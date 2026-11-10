"""SUNG-CQL adapted to the supplied CrowdNav experiment. Run as a module."""
import argparse
import configparser
import copy
import json
import logging
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch

from crowd_nav.sung_impl.conservative_sac import ConservativeSAC
from crowd_nav.sung_impl.model import TanhGaussianPolicy, FullyConnectedQFunction
from crowd_nav.sung_impl.utils import OORB
from crowd_nav.sung_impl.state import transform


def parse_args():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['both', 'offline', 'vae', 'online', 'test'], default='both')
    p.add_argument('--offline_dataset_path', default=str(root/'Crowd_nav_7'/'Crowd_nav_7.hdf5'))
    p.add_argument('--env_config', default=str(root/'configs'/'env.config'))
    p.add_argument('--env', default='CrowdSim-v0')
    p.add_argument('--checkpoints_path', default=str(root/'data'/'output_sung_cql_v5_attention_7'))
    p.add_argument('--checkpoint', default=None, help='Override input checkpoint in vae/online/test mode')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--state_dim', type=int, choices=[65,91], default=91, help='13 features per human: 7 humans = 91 (5 humans = 65)')
    p.add_argument('--action_dim', type=int, default=2)
    p.add_argument('--max_action_value', type=float, default=1.)
    p.add_argument('--offline_updates', '--offline_steps', dest='offline_steps', type=int, default=100000,
                   help='Minibatch update rounds, including BC warmup; VAE updates once per round')
    p.add_argument('--vae_steps', type=int, default=100000, help='Standalone --mode vae only')
    p.add_argument('--online_updates', '--online_steps', dest='online_steps', type=int, default=50000,
                   help='Minibatch update rounds, NOT environment interaction steps')
    p.add_argument('--online_episodes', type=int, default=0, help='Optional additional episode cap; 0 disables')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--updates_per_step', type=int, default=1)
    p.add_argument('--updates_per_episode', type=int, default=1, help='If positive, replaces per-step updates')
    p.add_argument('--eval_freq', type=int, default=1000)
    p.add_argument('--n_episodes', type=int, default=50)
    p.add_argument('--final_test_episodes', type=int, default=500,
                   help='Held-out test episodes for the final offline checkpoint; 0 disables')
    p.add_argument('--eval_phase', choices=['val','test'], default='val')
    p.add_argument('--log_freq', type=int, default=1000)
    p.add_argument('--save_freq', type=int, default=5000)
    p.add_argument('--discount', type=float, default=.99)
    p.add_argument('--tau', type=float, default=.005)
    p.add_argument('--actor_lr', type=float, default=1e-4)
    p.add_argument('--qf_lr', type=float, default=3e-4)
    p.add_argument('--policy_arch', default='256-256')
    p.add_argument('--encoder', choices=['attention','mlp'], default='attention', help='CrowdNav human encoder; 13 features per human')
    p.add_argument('--qf_arch', default='256-256')
    p.add_argument('--cql_min_q_weight', type=float, default=1., help='CrowdNav tuning; upstream default is 5')
    p.add_argument('--target_entropy', type=float, default=-4., help='CrowdNav tuning; upstream 2-D default is -2')
    p.add_argument('--initial_alpha', type=float, default=.1, help='Initial SAC entropy temperature')
    p.add_argument('--bc_entropy_weight', type=float, default=0., help='0: likelihood-only warmup; 1: upstream entropy-regularized BC')
    p.add_argument('--bc_steps', type=int, default=10000, help='Actor BC warmup within offline update budget; 0 disables')
    p.add_argument('--no_normalize_states', action='store_true', help='Disable fixed offline state statistics for actor/critics')
    p.add_argument('--candidate_num', type=int, default=100)
    p.add_argument('--topk', type=int, default=20)
    p.add_argument('--exploration', choices=['default','greedy','prob'], default='default')
    # p.add_argument('--exploration', choices=['default', 'greedy', 'prob'], default='prob')

    p.add_argument('--order', choices=['uq','qu'], default='uq')
    p.add_argument('--constrained', choices=['default','greedy','prob','all'], default='prob')
    p.add_argument('--lamda', type=float, default=1)
    # p.add_argument('--lamda', type=float, default=.1)
    # p.add_argument('--prob', type=float, default=.1)
    p.add_argument('--prob', type=float, default=0.5)
    # p.add_argument('--replay_p', type=float, default=.1, help='Fraction from global offline+online buffer')
    p.add_argument('--replay_p', type=float, default=1, help='Fraction from global offline+online buffer')
    p.add_argument('--global_buffer_size', type=int, default=3000000)
    p.add_argument('--online_buffer_max_size', type=int, default=20000)
    p.add_argument('--timeout_terminal', action='store_true', help='Match v17 done semantics instead of SUNG truncation bootstrap')
    a = p.parse_args()
    if a.action_dim != 2 or a.max_action_value != 1.:
        p.error('This CrowdNav adapter requires 2-D actions bounded by [-1,1], matching the supplied code.')
    if not 1 <= a.topk <= a.candidate_num or not 0 < a.prob <= 1 or int(a.batch_size*a.prob) < 1:
        p.error('Require 1 <= topk <= candidate_num and 1 <= int(batch_size*prob) <= batch_size.')
    if not 0 <= a.replay_p <= 1:
        p.error('replay_p must be in [0,1]')
    if min(a.batch_size,a.n_episodes,a.log_freq,a.save_freq,a.online_buffer_max_size) <= 0:
        p.error('batch/evaluation/log/save/buffer counts must be positive')
    if min(a.offline_steps,a.vae_steps,a.online_steps,a.updates_per_step,a.updates_per_episode,a.online_episodes) < 0:
        p.error('step counts cannot be negative')
    if a.bc_steps < 0 or a.final_test_episodes < 0 or (a.updates_per_episode == 0 and a.updates_per_step == 0):
        p.error('bc_steps must be nonnegative and online update scheduling must be enabled')
    if not 0 < a.discount <= 1 or not 0 < a.tau <= 1 or a.lamda < 0 or a.cql_min_q_weight < 0:
        p.error('Invalid discount, tau or CQL regularization weight')
    if not np.isfinite([a.initial_alpha,a.target_entropy,a.bc_entropy_weight]).all() or a.initial_alpha<=0 or a.bc_entropy_weight<0:
        p.error('Entropy settings must be finite; initial_alpha > 0 and bc_entropy_weight >= 0')
    return a


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(a):
    keys = ['observations','actions','next_observations','rewards','terminals']
    with h5py.File(a.offline_dataset_path, 'r') as f:
        missing = set(keys)-set(f.keys())
        if missing:
            raise ValueError(f'HDF5 missing keys: {missing}')
        d = {k: np.asarray(f[k], dtype=np.float32) for k in keys}
        if 'timeouts' in f:
            timeout = np.asarray(f['timeouts']).reshape(-1)
            if timeout.size not in (0, len(d['observations'])):
                raise ValueError('timeouts must be empty or have one flag per transition')
            if timeout.size:
                if not np.isin(timeout, [0, 1]).all():
                    raise ValueError('timeouts must be binary')
                d['terminals'] = d['terminals'].reshape(-1)
                d['terminals'][timeout.astype(bool)] = float(a.timeout_terminal)
            else:
                logging.warning('Empty timeouts: preserving dataset terminals; timeout labels cannot be inferred safely.')
    n = len(d['observations'])
    if n == 0: raise ValueError('Empty dataset')
    for k in keys:
        if len(d[k]) != n or not np.isfinite(d[k]).all():
            raise ValueError(f'Invalid length or non-finite values in {k}')
    for k, dim in [('observations',a.state_dim),('next_observations',a.state_dim),('actions',a.action_dim)]:
        if d[k].shape != (n, dim):
            raise ValueError(f'{k}: expected {(n,dim)}, got {d[k].shape}. Match human count and flattening to iql_rrnd.transform.')
    for k in ['rewards','terminals']:
        if d[k].size != n: raise ValueError(f'{k} must have one scalar per transition')
        d[k] = d[k].reshape(-1)
    if not np.isin(d['terminals'], [0.,1.]).all(): raise ValueError('terminals must be binary')
    if np.abs(d['actions']).max() > 1.00001: raise ValueError('Dataset actions exceed [-1,1]; verify action coordinates/scaling')
    # Match official SUNG CQL input action clipping. Rewards and policy states stay raw.
    d['actions'] = np.clip(d['actions'], -.99, .99)
    return d


def build(a, d):
    pi = TanhGaussianPolicy(a.state_dim,a.action_dim,arch=a.policy_arch,log_std_offset=-1.,encoder=a.encoder)
    q1 = FullyConnectedQFunction(a.state_dim,a.action_dim,arch=a.qf_arch,encoder=a.encoder)
    q2 = FullyConnectedQFunction(a.state_dim,a.action_dim,arch=a.qf_arch,encoder=a.encoder)
    sac = ConservativeSAC(dict(target_entropy=a.target_entropy, initial_alpha=a.initial_alpha,
        bc_entropy_weight=a.bc_entropy_weight, discount=a.discount,
        policy_lr=a.actor_lr,qf_lr=a.qf_lr,soft_target_update_rate=a.tau,
        cql_min_q_weight=a.cql_min_q_weight),pi,q1,q2,copy.deepcopy(q1),copy.deepcopy(q2),
        a.state_dim,a.action_dim,1.,device_override=a.device)
    sac.torch_to_device(a.device)
    # Update budgets do not bound environment steps in episode-based training.
    capacity = max(a.global_buffer_size,len(d['observations'])+1)
    replay = OORB(a.state_dim,a.action_dim,capacity,a.online_buffer_max_size,a.replay_p)
    replay.device = torch.device(a.device)
    replay.offline_buffer.device = replay.online_buffer.device = replay.device
    replay.convert_D4RL(d,shared_human_stats=a.encoder=='attention')
    if not a.no_normalize_states:
        for network in [pi,q1,q2,sac.target_qf1,sac.target_qf2]:
            network.observation_mean.copy_(replay.mean)
            network.observation_std.copy_(replay.std)
    return sac,replay


def save(path,sac,replay,a,stage,env_steps=0,stage_updates=0):
    payload = dict(format='sung_crowdnav_v5',config=vars(a),stage=stage,env_steps=env_steps,
        stage_updates=stage_updates,mean=replay.mean,std=replay.std,total_updates=sac.total_steps,
        modules=[m.state_dict() for m in sac.modules],
        optimizers={k:getattr(sac,k).state_dict() for k in ['policy_optimizer','qf_optimizer','alpha_optimizer','vae_optimizer']})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()
    logging.info('Checkpoint saved: %s (stage=%s, stage_updates=%d, env_steps=%d)',
                 path.resolve(), stage, stage_updates, env_steps)


def save_pretrain(out,sac,replay,a,stage,update,final=False):
    stem = 'sung_cql_pretrained' if stage == 'cql' else 'sung_vae'
    save(out/f'{stem}_{update:06d}.pt',sac,replay,a,stage,stage_updates=update)
    save(out/f'{stem}.pt',sac,replay,a,stage,stage_updates=update)
    if stage == 'vae' and final:
        save(out/'sung_offline.pt',sac,replay,a,'ready',stage_updates=update)


def load(path,sac,replay,a):
    try: ck = torch.load(path,map_location=a.device,weights_only=False)
    except TypeError: ck = torch.load(path,map_location=a.device)
    if ck.get('format') != 'sung_crowdnav_v5':
        raise ValueError('Requires a v5 checkpoint with matching human encoder and normalization; old files are preserved.')
    for k in ['state_dim','action_dim','encoder','policy_arch','qf_arch','timeout_terminal','discount','no_normalize_states','cql_min_q_weight','target_entropy','initial_alpha','bc_entropy_weight']:
        if ck['config'][k] != getattr(a,k): raise ValueError(f'Checkpoint config mismatch: {k}')
    for m,s in zip(sac.modules,ck['modules']): m.load_state_dict(s)
    for k,s in ck['optimizers'].items(): getattr(sac,k).load_state_dict(s)
    if not torch.allclose(replay.mean,ck['mean'],atol=1e-6) or not torch.allclose(replay.std,ck['std'],atol=1e-6):
        raise ValueError('Offline observation statistics differ from checkpoint')
    replay.mean,replay.std = ck['mean'],ck['std']
    sac._total_steps = ck['total_updates']
    return ck


def make_env(a,sac):
    import gym
    import crowd_sim  # environment registration
    from crowd_sim.envs.utils.robot import Robot
    conf = configparser.RawConfigParser()
    if not conf.read(a.env_config): raise FileNotFoundError(a.env_config)
    env = gym.make(a.env); env.configure(conf)
    robot = Robot(conf,'robot'); env.set_robot(robot)
    # CrowdNav Robot.set_policy uses policy.kinematics.
    sac.policy.kinematics = 'holonomic'
    # Match the supplied IQL actor: keep the configured multi-human scenario.
    sac.policy.multiagent_training = True
    robot.set_policy(sac.policy)
    return env,robot


def state_vector(observation,robot,a):
    from crowd_sim.envs.utils.state import JointState
    x = transform(JointState(robot.get_full_state(),observation),a.device).flatten().cpu().numpy()
    if x.shape != (a.state_dim,): raise ValueError(f'Environment state {x.shape} != ({a.state_dim},); check human count')
    return x


def evaluate(a,sac,step,out,stage='test',env_steps=0):
    from crowd_sim.envs.utils.action import ActionXY
    from crowd_sim.envs.utils.info import ReachGoal,Collision,Timeout
    # Isolate evaluation RNG so evaluations do not change training sampling.
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    env=None
    try:
        seed_all(a.seed+10000)
        env,robot=make_env(a,sac)
        counts=[0,0,0]; nav=[]; returns=[]; raw=[]
        for case in range(a.n_episodes):
            obs=env.reset(phase=a.eval_phase,test_case=case); done=False; ret=0.; undiscounted=0.; t=0
            while not done:
                action=sac.select_action(state_vector(obs,robot,a))
                obs,r,done,info=env.step(ActionXY(float(action[0]),float(action[1])))
                ret += a.discount**t*float(r); undiscounted += float(r); t+=1
            if isinstance(info,ReachGoal): counts[0]+=1; nav.append(float(env.global_time))
            elif isinstance(info,Collision): counts[1]+=1
            elif isinstance(info,Timeout): counts[2]+=1
            else: raise RuntimeError(f'Unknown terminal info: {info}')
            returns.append(ret); raw.append(undiscounted)
        metrics=dict(kind='eval',stage=stage,update=step,env_steps=env_steps,phase=a.eval_phase,success_rate=counts[0]/a.n_episodes,success=counts[0],collision=counts[1],timeout=counts[2],
            avg_nav_time=float(np.mean(nav)) if nav else None,avg_reward=float(np.mean(returns)),undiscounted_reward=float(np.mean(raw)),episodes=a.n_episodes)
        record(out,metrics)
        return metrics
    finally:
        if env is not None: env.close()
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None: torch.cuda.set_rng_state_all(rng[3])


def record(out,metrics):
    line=json.dumps(metrics,ensure_ascii=False,allow_nan=False)
    if metrics['kind']=='eval':
        logging.info('[%s] Success Rate: %.3f, Collision Rate: %.3f, Navtime: %.3f, Avg Reward: %.3f',
                     'TEST' if metrics['phase']=='test' else 'VAL',
                     metrics['success_rate'],metrics['collision']/metrics['episodes'],
                     metrics['avg_nav_time'] or 0.,metrics['avg_reward'])
    elif metrics['kind']=='online_episode':
        logging.info('[ONLINE_SUNG] ep=%d/%s | succ=%d | coll=%d | timeout=%d | nav_time=%.4f | reward=%.4f | buffer_size=%d',
                     metrics['episode'],metrics['episode_budget'] or '?',metrics['success'],
                     metrics['collision'],metrics['timeout'],metrics['nav_time'],
                     metrics['reward'],metrics['buffer_size'])
    else:
        logging.info(line)
    with open(out/'metrics.jsonl','a',encoding='utf-8') as f: f.write(line+'\n')


def pretrain_vae(a,sac,replay,out):
    for i in range(a.vae_steps):
        s,ac,*_=replay.sample(a.batch_size)
        loss=sac.vae.elbo_loss((s-replay.mean)/replay.std,ac,beta=.5).mean()
        sac.vae_optimizer.zero_grad(); loss.backward(); sac.vae_optimizer.step()
        if (i+1)%a.log_freq==0: record(out,dict(kind='vae',update=i+1,loss=loss.item()))
        if (i+1)%a.save_freq==0:
            save_pretrain(out,sac,replay,a,'vae',i+1)
    save_pretrain(out,sac,replay,a,'vae',a.vae_steps,final=True)


def prepare_online(sac):
    """Transfer networks and reset online optimizers/alpha using the configured initialization."""
    sac.update_target_network(1.)
    with torch.no_grad():
        sac.log_alpha.constant.fill_(np.log(sac.config.initial_alpha))
    for key in ['policy_optimizer','qf_optimizer','alpha_optimizer','vae_optimizer']:
        getattr(sac,key).state.clear()
    sac._total_steps = 0


def online(a,sac,replay,out):
    from crowd_sim.envs.utils.action import ActionXY
    from crowd_sim.envs.utils.info import ReachGoal,Collision,Timeout
    prepare_online(sac)
    sac.config.use_cql = a.constrained != 'default'
    evaluate(a,sac,0,out,stage='online')
    env,robot=make_env(a,sac)
    if hasattr(env,'seed'): env.seed(a.seed)
    obs=env.reset(phase='train'); ep=0; ret=0.; raw_ret=0.; ep_steps=0
    episode_budget=((a.online_steps+a.updates_per_episode-1)//a.updates_per_episode
                    if a.updates_per_episode else a.online_episodes)
    if a.online_episodes:
        episode_budget=min(episode_budget,a.online_episodes) if episode_budget else a.online_episodes
    step=0; update=0; best=-1.
    try:
        while update < a.online_steps:
            step+=1
            s=state_vector(obs,robot,a)
            ac=sac.select_exploration_action(s,a.exploration,a.order,a.candidate_num,a.topk,replay.mean,replay.std)
            obs,r,done,info=env.step(ActionXY(float(ac[0]),float(ac[1])))
            ns=state_vector(obs,robot,a)
            terminal=float(done and (a.timeout_terminal or not isinstance(info,Timeout)))
            replay.add(s,ac,ns,float(r),terminal)
            ret+=a.discount**ep_steps*float(r); raw_ret+=float(r); ep_steps+=1
            updates = (a.updates_per_episode if done else 0) if a.updates_per_episode else a.updates_per_step
            for _ in range(min(updates,a.online_steps-update)):
                last=sac.train(replay,a.batch_size,a.constrained,a.lamda,a.prob)
                update+=1
                if update%a.log_freq==0:
                    record(out,dict(kind='online_update',update=update,env_steps=step,**last))
                if a.eval_freq>0 and update%a.eval_freq==0:
                    metrics=evaluate(a,sac,update,out,stage='online',env_steps=step)
                    if metrics['success_rate'] > best:
                        best=metrics['success_rate']
                        save(out/'sung_online_best.pt',sac,replay,a,'online',step,update)
                if update%a.save_freq==0:
                    save(out/'sung_online.pt',sac,replay,a,'online',step,update)
            if done:
                ep+=1
                record(out,dict(kind='online_episode',update=update,env_steps=step,episode=ep,length=ep_steps,reward=ret,
                    undiscounted_reward=raw_ret,episode_budget=episode_budget,
                    nav_time=float(env.global_time) if isinstance(info,ReachGoal) else 0.,
                    buffer_size=replay.online_buffer.size,global_buffer_size=replay.offline_buffer.size,
                    success=int(isinstance(info,ReachGoal)),collision=int(isinstance(info,Collision)),timeout=int(isinstance(info,Timeout))))
                obs=env.reset(phase='train'); ret=0.; raw_ret=0.; ep_steps=0
            if a.online_episodes and ep>=a.online_episodes: break
        if update < a.online_steps:
            logging.warning('Episode cap stopped training at %d/%d updates',update,a.online_steps)
        if a.eval_freq<=0 or update%a.eval_freq or update==0:
            evaluate(a,sac,update,out,stage='online',env_steps=step)
        save(out/'sung_online.pt',sac,replay,a,'online',step,update)
    finally: env.close()


def main():
    a=parse_args(); seed_all(a.seed)
    out=Path(a.checkpoints_path); out.mkdir(parents=True,exist_ok=True)
    log_name='online.log' if a.mode=='online' else 'train.log'
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/log_name,encoding='utf-8')])
    (out/f'config_{a.mode}.json').write_text(json.dumps(vars(a),indent=2),encoding='utf-8')
    d=load_dataset(a); sac,replay=build(a,d)
    record(out,dict(kind='dataset',size=len(d['observations']),state_dim=a.state_dim,action_dim=a.action_dim,mode=a.mode))
    if a.mode in ['offline','both']:
        best=-1.
        for i in range(a.offline_steps):
            # Algorithm 1: one actor/critic update AND one density update per round.
            m=sac.train(replay,a.batch_size,strategy='all',lamda=1.,prob=1.,bc=i<a.bc_steps,update_vae=True)
            if (i+1)%a.log_freq==0: record(out,dict(kind='offline',update=i+1,**m))
            if (a.eval_freq>0 and (i+1)%a.eval_freq==0) or i+1==a.offline_steps:
                metrics=evaluate(a,sac,i+1,out,stage='offline')
                if metrics['success_rate'] > best:
                    best=metrics['success_rate']
                    save(out/'sung_offline_best.pt',sac,replay,a,'ready',stage_updates=i+1)
            if (i+1)%a.save_freq==0:
                save_pretrain(out,sac,replay,a,'cql',i+1)
        save_pretrain(out,sac,replay,a,'cql',a.offline_steps,final=True)
        save(out/'sung_offline.pt',sac,replay,a,'ready',stage_updates=a.offline_steps)
        if a.final_test_episodes:
            test_args=copy.copy(a)
            test_args.eval_phase='test'; test_args.n_episodes=a.final_test_episodes
            result=evaluate(test_args,sac,a.offline_steps,out,stage='offline_final')
            (out/'offline_final_result.json').write_text(
                json.dumps(dict(result,target_success_rate=.6,
                                target_met=result['success_rate']>=.6),indent=2),encoding='utf-8')
    elif a.mode=='vae':
        load(a.checkpoint or out/'sung_cql_pretrained.pt',sac,replay,a)
        pretrain_vae(a,sac,replay,out)
    else:
        ck=load(a.checkpoint or out/'sung_offline.pt',sac,replay,a)
        if a.mode=='online' and ck['stage']!='ready':
            raise ValueError('Online starts a fresh run from stage=ready. Replay/environment resume is not implemented.')
    if a.mode in ['both','online']: online(a,sac,replay,out)
    elif a.mode=='test': evaluate(a,sac,ck['stage_updates'],out,stage=ck['stage'],env_steps=ck['env_steps'])


if __name__=='__main__': main()

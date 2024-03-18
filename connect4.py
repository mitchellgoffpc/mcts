import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.2'

import chex
import mctx
import jax
import jax.numpy as jnp
import functools
import rich

Board = chex.Array
Action = chex.Array # Index of the column to play
Player = chex.Array  # 1 if player X, -1 if player O
Reward = chex.Array  # 1 for winning, 0 for draw, -1 for losing
Done = chex.Array  # True/False if the game is over

@chex.dataclass
class Env:
    board: Board
    player: Player
    done: Done
    reward: Reward


BOARD_STRING = """
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 1   2   3   4   5   6   7
"""

def print_board(board: Board):
    board_str = BOARD_STRING
    for i in reversed(range(board.shape[0])):
        for j in range(board.shape[1]):
            board_str = board_str.replace('?', '[green]X[/green]' if board[i, j] == 1 else '[red]O[/red]' if board[i, j] == -1 else ' ', 1)
    rich.print(board_str)


# Environment dynamics

def horizontals(board: Board) -> chex.Array:
    return jnp.stack([
        board[i, j:j+4]
        for i in range(board.shape[0])
        for j in range(board.shape[1] - 3)
    ])

def verticals(board: Board) -> chex.Array:
    return jnp.stack([
        board[i:i+4, j]
        for i in range(board.shape[0] - 3)
        for j in range(board.shape[1])
    ])

def diagonals(board: Board) -> chex.Array:
    return jnp.stack([
        jnp.diag(board[i:i+4, j:j+4])
        for i in range(board.shape[0] - 3)
        for j in range(board.shape[1] - 3)
    ])

def antidiagonals(board: Board) -> chex.Array:
    return jnp.stack([
        jnp.diag(board[i:i+4, j:j+4][::-1])
        for i in range(board.shape[0] - 3)
        for j in range(board.shape[1] - 3)
    ])

def get_winner(board: Board) -> Player:
    all_lines = jnp.concatenate((
        horizontals(board),
        verticals(board),
        diagonals(board),
        antidiagonals(board),
    ))
    x_won = jnp.any(jnp.all(all_lines == 1, axis=1)).astype(jnp.int8)
    o_won = jnp.any(jnp.all(all_lines == -1, axis=1)).astype(jnp.int8)
    return x_won - o_won

def env_reset(_):
    return Env(
        board=jnp.zeros((6, 7), dtype=jnp.int8),
        player=jnp.int8(1),
        done=jnp.bool_(False),
        reward=jnp.int8(0))

def env_step(env: Env, action: Action) -> tuple[Env, Reward, Done]:
    col = action
    row = jnp.argmax(env.board[:, col] == 0)  # find the first empty row in the column
    invalid_move = env.board[row, col] != 0  # if the column is full, the move is invalid
    # Place the player's piece in the board only if the move is valid and the game is not over.
    board = env.board.at[row, col].set(jnp.where(env.done | invalid_move, env.board[row, col], env.player))
    reward = jnp.where(env.done, 0, jnp.where(invalid_move, -1, get_winner(board) * env.player)).astype(jnp.int8)
    done = env.done | (reward != 0) | invalid_move | jnp.all(board[-1] != 0)
    env = Env(
        board=board,
        player=jnp.where(done, env.player, -env.player),  # alternate players
        done=done,
        reward=reward)

    return env, reward, done


# Policy and value functions

def valid_action_mask(env: Env) -> chex.Array:
    return jnp.where(env.done, jnp.array([False] * env.board.shape[1]), env.board[-1] == 0)

def winning_action_mask(env: Env, player: Player) -> chex.Array:
    env = Env(board=env.board, player=player, done=env.done, reward=env.reward)
    env, reward, done = jax.vmap(env_step, (None, 0))(env, jnp.arange(7, dtype=jnp.int8))
    return reward == 1

def policy_function(env: Env) -> chex.Array:
    return sum((
        valid_action_mask(env).astype(jnp.float32) * 100,
        winning_action_mask(env, -env.player).astype(jnp.float32) * 200,
        winning_action_mask(env, env.player).astype(jnp.float32) * 300,
    ))

def value_function(env: Env, rng_key: chex.PRNGKey) -> Reward:
    # Run a rollout to the end of the game and return the reward.
    def cond(a):
        env, key = a
        return ~env.done
    def step(a):
        env, key = a
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, policy_function(env))
        env, reward, done = env_step(env, action)
        return env, key
    leaf, key = jax.lax.while_loop(cond, step, (env, rng_key))
    # The leaf reward is from the perspective of the last player.
    # We negate it if the last player is not the initial player.
    return (leaf.reward * leaf.player * env.player).astype(jnp.float32)


# MCTS dynamics

def root_fn(env: Env, rng_key: chex.PRNGKey) -> mctx.RootFnOutput:
    return mctx.RootFnOutput(
        prior_logits=policy_function(env),
        value=value_function(env, rng_key),
        embedding=env)  # We will use the `embedding` field to store the environment.

def recurrent_fn(params, rng_key, action, embedding) -> tuple[mctx.RecurrentFnOutput, Env]:
    env = embedding
    env, reward, done = env_step(env, action)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(done, 0, -1).astype(jnp.float32),
        prior_logits=policy_function(env),
        value=jnp.where(done, 0, value_function(env, rng_key)).astype(jnp.float32))

    return recurrent_fn_output, env

@functools.partial(jax.jit, static_argnums=(2,))
def run_mcts(rng_key: chex.PRNGKey, env: Env, num_simulations: int) -> chex.Array:
    batch_size = 1
    key1, key2 = jax.random.split(rng_key)
    return mctx.muzero_policy(
        params=None,
        rng_key=key1,
        root=jax.vmap(root_fn, (None, 0))(env, jax.random.split(key2, batch_size)),
        recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
        num_simulations=num_simulations,
        max_depth=42,
        qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),
        dirichlet_fraction=0.0)


if __name__ == '__main__':
    batch_size = 128
    key = jax.random.PRNGKey(0)
    envs = jax.vmap(env_reset)(jax.numpy.arange(batch_size))

    import time
    start_time = time.time()

    while jnp.any(~envs.done):
        key, skey = jax.random.split(key)
        rng_keys = jax.random.split(skey, batch_size)
        policy_outputs = jax.vmap(run_mcts, in_axes=(0, 0, None))(rng_keys, envs, 1000)
        actions = jax.vmap(lambda key, p: jax.random.choice(key, 7, p=p))(rng_keys, policy_outputs.action_weights.squeeze(1))
        envs, _, _ = jax.vmap(env_step, in_axes=(0, 0))(envs, actions)
        print(jnp.sum(envs.done))

    end_time = time.time()
    print(f"Duration: {end_time - start_time} seconds")

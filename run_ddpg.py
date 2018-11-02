import ddpg.train as train


def run_ddpg_simple():
    train.train_ddpg_loop(
        run_name="test_run_simple_03",
        # run_name="test_run_all_together_better_norm",
        # run_name="test_run_all_together_better_norm_with_reg",
        seed=2020,

        env_normalize_observations=True,
        env_normalize_returns=True,

        critic_l2_reg=0.0,
        ops_together=False,

        roller_normalize_observations=False,
        roller_normalize_returns=False
    )


def run_ddpg_simple_better_norm():
    train.train_ddpg_loop(
        run_name="test_run_simple_better_norm_03",
        # run_name="test_run_all_together_better_norm",
        # run_name="test_run_all_together_better_norm_with_reg",
        seed=2020,

        env_normalize_observations=False,
        env_normalize_returns=False,

        critic_l2_reg=0.0,
        ops_together=False,

        roller_normalize_observations=True,
        roller_normalize_returns=True
    )


def run_ddpg_all_together():
    train.train_ddpg_loop(
        run_name="test_run_all_together_02",
        seed=2019,

        env_normalize_observations=True,
        env_normalize_returns=True,

        critic_l2_reg=0.0,
        ops_together=True,

        roller_normalize_observations=False,
        roller_normalize_returns=False
    )


def run_ddpg_all_together_better_normalization():
    train.train_ddpg_loop(
        run_name="test_run_all_together_better_normalization_03",
        seed=2020,

        env_normalize_observations=False,
        env_normalize_returns=False,

        critic_l2_reg=0.0,
        ops_together=True,

        roller_normalize_observations=True,
        roller_normalize_returns=True
    )


def run_ddpg_all_together_better_normalization_with_reg():
    train.train_ddpg_loop(
        run_name="test_run_all_together_better_normalization_with_reg_03",
        seed=2019,

        env_normalize_observations=False,
        env_normalize_returns=False,

        critic_l2_reg=0.01,
        ops_together=True,

        roller_normalize_observations=True,
        roller_normalize_returns=True
    )


if __name__ == '__main__':
    # run_ddpg_simple()
    # run_ddpg_simple_better_norm()
    # run_ddpg_all_together()
    run_ddpg_all_together_better_normalization()
    # run_ddpg_all_together_better_normalization_with_reg()

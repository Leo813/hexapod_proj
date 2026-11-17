"""Application lifecycle around the Stage 2 algorithm, with no physics changes."""

from pathlib import Path

from .checkpoints import load_checkpoint, save_checkpoint
from .collector import RolloutCollector
from .integration.configuration import TrainingConfig


class TrainingLogger:
    def __init__(self, print_stats=True, tensorboard_dir=None):
        self.print_stats = print_stats
        self.writer = None
        if tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(tensorboard_dir))

    def log(self, metrics):
        if self.print_stats:
            print(" ".join(f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
                           for key, value in metrics.items()), flush=True)
        if self.writer is not None:
            for key, value in metrics.items():
                if value is not None:
                    self.writer.add_scalar(key, value, metrics["frames"])

    def close(self):
        if self.writer is not None:
            self.writer.close()


class Trainer:
    def __init__(self, algorithm, environment, output_dir, config=None, logger=None, resolved_config=None):
        self.algorithm, self.environment = algorithm, environment
        self.config = config if config is not None else TrainingConfig()
        self.output_dir = Path(output_dir)
        self.logger = logger if logger is not None else TrainingLogger(self.config.print_stats)
        self.resolved_config = resolved_config
        self.epoch = self.frames = 0
        self.best_reward = -100500.0  # reference initial last_mean_rewards
        self.collector = None

    def counters(self):
        return dict(epoch=self.epoch, frames=self.frames, best_reward=self.best_reward)

    def save(self, filename):
        save_checkpoint(self.output_dir / "checkpoints" / filename, self.algorithm, self.counters(),
                        self.config, self.resolved_config)

    def resume(self, path, trust_legacy=False):
        if self.collector is not None:
            raise RuntimeError("Resume before collection, using a fresh simulator environment")
        counters = load_checkpoint(path, self.algorithm, resume=True, trust_legacy=trust_legacy)
        self.epoch, self.frames, self.best_reward = counters["epoch"], counters["frames"], counters["best_reward"]

    def run(self):
        """Epoch/frame counts include only completed PPO updates; always clean up."""
        history = []
        try:
            if self.config.max_epochs != -1 and self.epoch >= self.config.max_epochs:
                return history
            self.collector = RolloutCollector(self.algorithm, self.environment, self.config.games_to_track)
            while self.config.max_epochs == -1 or self.epoch < self.config.max_epochs:
                self.environment.set_train_info(self.frames)
                rollout = self.collector.collect()
                updates = self.algorithm.update(rollout)
                self.epoch += 1
                self.frames += self.algorithm.config.batch_size
                stats = self.collector.statistics.summary()
                metrics = dict(epoch=self.epoch, frames=self.frames, **stats)
                for key in ("policy", "critic", "entropy", "bounds", "total", "kl"):
                    metrics[key] = sum(update[key] for update in updates) / len(updates)
                metrics["learning_rate_start"] = updates[0]["learning_rate"]
                metrics["learning_rate"] = self.algorithm.learning_rate
                self.logger.log(metrics)
                history.append(metrics)
                reward = stats["episode_reward"]
                won = False
                if reward is not None:
                    if (self.config.save_frequency > 0 and self.epoch % self.config.save_frequency == 0
                            and reward <= self.best_reward):
                        self.save(f"epoch_{self.epoch:06d}.pt")
                    if reward > self.best_reward and self.epoch >= self.config.save_best_after:
                        self.best_reward = reward
                        self.save("best.pt")
                        won = self.best_reward > self.config.score_to_win
                        if won:
                            self.save("winning.pt")
                if won or (self.config.max_epochs != -1 and self.epoch >= self.config.max_epochs):
                    self.save("last.pt")
                    break
            return history
        finally:
            try:
                self.environment.close()
            finally:
                self.logger.close()

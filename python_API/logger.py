import tensorflow as tf
import os

log_dir = "logs"

def create_logger():
    writer = tf.summary.create_file_writer(log_dir)
    def log(episode, reward):
        with writer.as_default():
            tf.summary.scalar("Total Reward", reward, step=episode)
    return log

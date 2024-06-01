import tensorflow_io as tfio
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import librosa


class Main:
    @staticmethod
    def main() -> None:
        print(f"tensorflow-io version: {tfio.version.VERSION}")
        print(f"tensorflow- version: {tf.version.VERSION}")
        print(f"numpy version: {np.version.version}")
        print(f"pandas version: {pd.__version__}")
        print(f"matplotlib version: {matplotlib.__version__}")
        print(f"librosa version: {librosa.version.version}")


if __name__ == "__main__":
    Main.main()

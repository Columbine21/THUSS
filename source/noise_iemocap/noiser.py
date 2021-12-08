import argparse
import sox
import numpy as np
from pathlib import Path
from tqdm import tqdm
seed = 1111
np.random.seed(seed)

class NoiseMaker():

    def __init__(self, n_dir, o_dir, t_dir, noise_type='NPARK', noise_level=1) -> None:
        self.__dict__.update(locals())

        noise_dir = Path(n_dir) / f"{noise_type}"
        self.noisy_wav_list = list(noise_dir.rglob('*.wav'))
        self.origin_wav_list = Path(o_dir).rglob('*.wav')

    def __call__(self) -> None:
        print(f"Begin to Processing the Noisy Dataset with Noise {self.noise_type} and Level {self.noise_level}.")
        print("*" * 100)
        for o in tqdm(self.origin_wav_list):
            d = sox.file_info.duration(o)
            origin = sox.Transformer().build_array(input_filepath=str(o))
            r_start = np.random.uniform(low=0, high=300-d)
            tfm = sox.Transformer().trim(r_start, r_start + d)
            noise = tfm.build_array(input_filepath=str(np.random.choice(self.noisy_wav_list)))
            output_dir = Path(self.t_dir).joinpath(f"{self.noise_type}-{self.noise_level}")
            output_dir.mkdir(exist_ok=True)
            tfm = sox.Transformer()
            tfm.build_file(
                input_array=noise*self.noise_level+origin, sample_rate_in=16000,
                output_filepath=str(output_dir.joinpath(f"{o.stem}_n.wav"))
            )
        print("*" * 100)
        print(f"NoiseMarker Finished!")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_dir', type=str, default="source/demand")
    parser.add_argument('--o_dir', type=str, default='source/iemocap/Audio_16k')
    parser.add_argument('--t_dir', type=str, default='source/noise_iemocap')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    noiser = NoiseMaker(args.n_dir, args.o_dir, args.t_dir, noise_level=2)()
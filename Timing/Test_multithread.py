from joblib import Parallel, delayed
import subprocess
import argparse
import time
from structlog import get_logger
import openfhe
from openfhe import SecretKeyDist, ScalingTechnique, CCParamsCKKSRNS, SecurityLevel, GenCryptoContext, PKESchemeFeature
from enc_GenHECNN import enc_GenHECNN

def run_exp(arg):
    command = "python3 exp_multithread.py"
    subprocess.run(command, shell=True)
    
if __name__ == "__main__":
    
    log = get_logger()

    parser = argparse.ArgumentParser(description='Launch the experiments.')
    parser.add_argument('--n_jobs', type=int, required=True, help='Number of parallel jobs: 1 deactivates multiprocessing.')
    parser.add_argument('--n_samples', type=int, required=True, help='Number of experiments to run for each case.')
    parser.add_argument('--n_experiments', type=int, required=False, default=1, help='Number of experiments to run for each case.')
    
    args = parser.parse_args()
    
    for j in range(args.n_experiments):
        
        log.info(f'Begin experiment for {args.n_samples} samples with {args.n_jobs} thread/s.\n')
        
        log.info('The context will be generated once at the beginning of the program.\n')
        log.info('Generating the context...')
        secret_key_dist = SecretKeyDist.UNIFORM_TERNARY
        rescale_tech = ScalingTechnique.FIXEDAUTO
        ring_dim = 2 ** 15
        dcrt_bits = 50
        first_mod = 60
        num_slots = ring_dim // 2

        parameters = CCParamsCKKSRNS()
        parameters.SetSecretKeyDist(secret_key_dist)
        parameters.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
        parameters.SetRingDim(ring_dim)
        parameters.SetScalingModSize(dcrt_bits)
        parameters.SetScalingTechnique(rescale_tech)
        parameters.SetFirstModSize(first_mod)
        parameters.SetBatchSize(num_slots)

        depth = 5
        parameters.SetMultiplicativeDepth(depth)
        cc = GenCryptoContext(parameters)
        log.info(f"The CKKS scheme is using a ring dimension of {cc.GetRingDimension()}.\n")

        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.FHE)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)
        cc.Enable(PKESchemeFeature.ADVANCEDSHE)

        log.info("Generating keys...")

        keys_time = time.time()

        key_pair = cc.KeyGen()
        cc.EvalMultKeyGen(key_pair.secretKey) # Relinearization key
        cc.EvalSumKeyGen(key_pair.secretKey) # Sum over all elements key

        keys_time = time.time() - keys_time
        log.info(f'Generating keys took: {keys_time} seconds.\n')
        
        log.info('Serializing context and keys...')
        if not openfhe.SerializeToFile("cc.txt", cc, openfhe.BINARY):
            raise Exception("Exception writing public key to cc.txt")
        log.info("Context have been serialized.")
        
        if not cc.SerializeEvalMultKey('multKey.txt', openfhe.BINARY):
            raise Exception("Error writing eval mult keys to multkeys.txt")
        log.info("EvalMult keys have been serialized.")
        
        if not openfhe.SerializeToFile("public_key.txt", key_pair.publicKey, openfhe.BINARY):
            raise Exception("Exception writing public key to public_key.txt")
        log.info("Public key have been serialized.")
        
        if not openfhe.SerializeToFile("secret_key.txt", key_pair.secretKey, openfhe.BINARY):
            raise Exception("Exception writing public key to secret_key.txt")
        log.info("Secret key have been serialized.\n")
        
        log.info("Begin experiments...\n")
        
        counter = time.time()
        
        Parallel(n_jobs=args.n_jobs)(delayed(run_exp)(i) for i in range(args.n_samples))
        
        counter = time.time() - counter
        
        log.info(f'Total time: {counter} seconds.')
        
        with open('HundredSamplesTime.txt', 'a') as file:
            file.write(str(counter) + '\n')
    
        log.info('-------------------------------------------------------------------------------\n\n')

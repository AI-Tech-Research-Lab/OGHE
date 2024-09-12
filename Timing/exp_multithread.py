import torch
import torch.nn.functional as FPPDL4CancerClassification
from structlog import get_logger
import pickle
import numpy as np
from scipy.linalg import block_diag
import time
from experiment_cnn_shuffle import GenomicCNN


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


if __name__ == '__main__':
            
    import openfhe
    from openfhe import GenCryptoContext
    from enc_GenHECNN import enc_GenHECNN
    
    
    log = get_logger()
    
    log.info('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    
    totalTime = time.time()
    
    CNV_size = 1024
    SNV_size = 2286
    n_labels = 11
    
    MODEL_PATH = f"cnn_{CNV_size}_{SNV_size}.pt"
    SAMPLE_DATA_PATH = f"sample_data_single_{CNV_size}_{SNV_size}.pkl"
    
    log.info('Loading model and data...')
    cnn = torch.load(MODEL_PATH)
    cnn.eval()
    
    log.debug('CNN characteristics:\n', cnn=cnn)

    with open(SAMPLE_DATA_PATH, 'rb') as f:
        batch = pickle.load(f)

    x = batch[0][0]
    x = x.view(1, len(x))
    
    # Load context
    log.info('Load context...')    
    openfhe.ReleaseAllContexts()

    cc, res = openfhe.DeserializeCryptoContext('cc.txt', openfhe.BINARY)
    if not res:
        raise Exception(
            "Cannot deserialize the cryptocontext from cc.txt"
        )
    log.info('Context has been loaded.')
    
    if not cc.DeserializeEvalMultKey('multKey.txt', openfhe.BINARY):
        raise Exception(
            "Cannot deserialize eval mult keys from multKey.txt"
        )
    log.info('EvalMult keys have been loaded.')
    
    public_key, res = openfhe.DeserializePublicKey('public_key.txt', openfhe.BINARY)
    if not res:
        raise Exception(
            "Cannot deserialize the public key from public_key.txt"
        )
    log.info('Public key has been loaded.')
    
    secret_key, res = openfhe.DeserializePrivateKey('secret_key.txt', openfhe.BINARY)
    if not res:
        raise Exception(
            "Cannot deserialize the public key from secret_key.txt"
        )
    log.info('Secret key has been loaded.\n')
    
    key_pair = [public_key, secret_key]
    
    log.info("Encoding model ...\n")
    encModel = enc_GenHECNN(cnn, cc, key_pair, [CNV_size, SNV_size])
    encModel.encodeModel()

    log.info('Encrypting model and data...')
    enc_x_CNV = encModel.dataEncoding(x[0, :CNV_size], 0)
    enc_x_SNV = encModel.dataEncoding(x[0, CNV_size:], 1)

    Totaltime_encryption = time.time()
    time_DataEncryption = time.time()
    
    ct_X_CNV = cc.Encrypt(public_key, cc.MakeCKKSPackedPlaintext(enc_x_CNV))
    ct_X_SNV = cc.Encrypt(public_key, cc.MakeCKKSPackedPlaintext(enc_x_SNV))
    
    time_DataEncryption = time.time() - time_DataEncryption
    time_ModelEncryption = time.time()
    
    encModel.encryptModel()
    
    time_ModelEncryption = time.time() - time_ModelEncryption

    Totaltime_encryption = time.time() - Totaltime_encryption
    
    log.info(f'Data encryption took {time_DataEncryption} seconds.\n')
    
    with open('M_DataEncrytionTime.txt', 'a') as file:
        file.write(str(time_DataEncryption) + '\n')
    
    log.info(f'Model encryption took {time_ModelEncryption} seconds.\n')
    
    with open('M_ModelEncrytionTime.txt', 'a') as file:
        file.write(str(time_ModelEncryption) + '\n')
        
    log.info(f'General encryption took {Totaltime_encryption} seconds.\n')
    
    with open('M_TotalEncrytionTime.txt', 'a') as file:
        file.write(str(Totaltime_encryption) + '\n')

    log.info('Starting the encrypted processing...')
    encProcessing_time = time.time()

    ct_pred_output = encModel.forward(ct_X_CNV, ct_X_SNV)

    encProcessing_time = time.time() - encProcessing_time
    log.info(f'Encrypted processing took {encProcessing_time} seconds.\n')
    
    with open('M_EncComputationTime.txt', 'a') as file:
        file.write(str(encProcessing_time) + '\n')

    log.info('Decrypting data...')
    decryption_time = time.time()

    obtained_result = cc.Decrypt(key_pair[1], ct_pred_output)
    obtained_result.SetLength(next_power_of_2(n_labels))

    decryption_time = time.time() - decryption_time
    log.info(f'Data decryption took {decryption_time} seconds.\n')
    
    with open('M_DecrytionTime.txt', 'a') as file:
        file.write(str(decryption_time) + '\n')

    # log.info('Comparison with the expected result...')
    # expected_output = cnn(x).detach().numpy()[0]

    # log.debug('Obtained output:', obtained_result=obtained_result)
    # log.debug('Expected output:', expected_output=expected_output)
    
    totalTime = time.time() - totalTime
    
    log.info(f'Total encrypted processing time per sample: {totalTime} seconds.')
    
import numpy as np
import pdb
from numpy import fft
from single_inference import inference

def per_pkt_transmission(args, MM, TransmittedSymbols):
    # Pass the channel and generate samples at the receiver

    # The taus are only for signal noise calculation
    taus = np.sort(np.random.uniform(0,args.maxDelay,(1,MM-1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(MM)
    for idx in np.arange(MM):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == MM-1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx-1]

    # # # Generate the channel: phaseOffset = 0->0; 1->2pi/4; 2->2pi/2; 3->2pi
    # np.random.seed(1896)
    if args.phaseOffset == 0:
        hh = np.ones([MM,1])
    elif args.phaseOffset == 1:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 2* np.pi/4)
    elif args.phaseOffset == 2:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 3* np.pi/4)
    else:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 4* np.pi /4)
    # print(hh)   # debug
    # complex pass the complex channel
    # 1. Use Matrix Mul Mythod
    LL = len(TransmittedSymbols[0])
    hh_expand = np.tile(hh,(1,TransmittedSymbols.shape[1]))
    TMAT1 = TransmittedSymbols*hh_expand
    d_data_vec = TMAT1.T.reshape(-1,1)
    D_array = np.zeros((MM*LL+MM-1,MM*LL))
    for idx in range(MM*LL):    # 按列依次赋值
        D_array[np.arange(MM)+idx, idx] = 1.
    samples1 = D_array @ d_data_vec
    samples = samples1.flatten()

    # 2. author's method
    for idx in range(MM):
        TransmittedSymbols[idx,:] = TransmittedSymbols[idx,:] * hh[idx][0]
    # print('Phase Misalign signal:',TransmittedSymbols[:,0])  # debug
    # compute the received signal power and add noise
    LL = len(TransmittedSymbols[0])
    SignalPart = np.sum(TransmittedSymbols,0)
    SigPower = np.sum(np.power(np.abs(SignalPart),2))/LL
    # SigPower = np.max(np.power(np.abs(SignalPart),2))
    EsN0 = np.power(10, args.EsN0dB/10.0)
    noisePower = SigPower/EsN0

    # # Oversample the received signal(might cause error)
    # RepeatedSymbols = np.repeat(TransmittedSymbols, MM, axis = 1)
    # for idx in np.arange(MM):
    #     extended = np.array([np.r_[np.zeros(idx), RepeatedSymbols[idx], np.zeros(MM-idx-1)]])
    #     if idx == 0:
    #         samples = extended
    #     else:
    #         samples = np.r_[samples, extended]
    # samples = np.sum(samples, axis=0)
    # print('difference:',np.sum(np.abs(samples1-samples))) # debug
    
    # generate noise
    for idx in np.arange(MM):
        noise = np.random.normal(loc=0, scale=np.sqrt(noisePower/2/dd[idx]), size=LL+1)+1j*np.random.normal(loc=0, scale=np.sqrt(noisePower/2/dd[idx]), size=LL+1)
        if idx == 0:
            AWGNnoise = np.array([noise])
        else:
            AWGNnoise = np.r_[AWGNnoise, np.array([noise])]

    AWGNnoise = np.reshape(AWGNnoise, (1,MM*(LL+1)), 'F')
    # samples = samples + AWGNnoise[0][0:-1]    # noise algorithm 1

    # noise algorithm 2
    def awgn(x, snr):
        '''Add AWGN
        x: numpy array
        snr: int, dB
        '''
        len_x = x.flatten().shape[0]
        Ps = np.sum(np.power(x, 2)) / len_x
        Pn = Ps / (np.power(10, snr / 10))
        noise = np.random.randn(x.shape[0]) * np.sqrt(Pn)/2 + 1j * (np.random.randn(x.shape[0]) * np.sqrt(Pn)/2)
        return x + noise
    samples = awgn(samples,args.EsN0dB)

    # np.save('orimethod_samples.npy',samples.flatten())  # debug
    outputs = [[],[],[],[],[],[]]
    if args.Estimator == -1:
    # aligned_sample estiamtor
    # if args.Estimator == 1:
        MthFiltersIndex = (np.arange(LL) + 1) * MM - 1
        output = samples.copy()[MthFiltersIndex]
        # print('For estimation:',output[0])# debug
        # return output
        outputs[0] = output.copy()

    # ML estiamtor
    # if args.Estimator == 2:
        noisePowerVec = noisePower/2./dd
        HH = np.zeros([MM*(LL+1)-1, MM*LL])*(1+0j)
        for idx in range(MM*LL):    # 按列依次赋值
            HH[np.arange(MM)+idx, idx] = hh[np.mod(idx,MM)]
        CzVec = np.tile(noisePowerVec, [1, LL+1])
        Cz = np.diag(CzVec[0][:-1])
        # Cz = np.diag(CzVec[0][:-1]/CzVec[0][:-1])
        ## ------------------------------------- ML
        MUD = np.matmul(HH.conj().T, np.linalg.inv(Cz))
        MUD = np.matmul(MUD, HH)
        MUD = np.matmul(np.linalg.inv(MUD), HH.conj().T)
        MUD = np.matmul(MUD, np.linalg.inv(Cz))
        MUD = np.matmul(MUD, np.array([samples]).T)
        # print('recover signal:',MUD.shape)  #debug
        ## ------------------------------------- Estimate SUM
        output = np.sum(np.reshape(MUD, [LL,MM]), 1)
        # return output
        outputs[1] = output.copy()

    # ML estiamtor
    # if args.Estimator == 41:
        import scipy.sparse.linalg as srlg
        from scipy.sparse import csr_matrix
        noisePowerVec = noisePower/2./dd
        HH = np.zeros([MM*(LL+1)-1, MM*LL]) * (1+0j)
        for idx in range(MM*LL):    # 按列依次赋值
            HH[np.arange(MM)+idx, idx] = hh[np.mod(idx,MM)]
        # print(HH[0:7,0:7])  # debug
        CzVec = np.tile(noisePowerVec, [1, LL+1])
        Cz = np.diag(CzVec[0][:-1])
        ## ------------------------------------- ML
        D_array_csr = csr_matrix(HH)               # 稀疏矩阵求解：5000x5000~2.7s
        d_inv_csr = srlg.inv(D_array_csr.conj().T@D_array_csr)
        restored = d_inv_csr @ D_array_csr.conj().T @ samples.T  

        ## ------------------------------------- Estimate SUM
        output = np.sum(np.reshape(restored, [LL, MM]), 1)
        # return output
        outputs[2] = output.copy()

    # SP_ML estiamtor
    # if args.Estimator == 3:
        noisePowerVec = noisePower/2./dd
        output = BP_Decoding(samples, MM, LL, hh, noisePowerVec)
        # return output
        outputs[3] = output.copy()
    
    # wiener estimator
    # if args.Estimator == 4:
        pad_len = len(samples)
        # print(samples)  # debug
        # print(MM,hh.shape)    # debug
        kernel = np.zeros_like(samples.flatten())   # 将信道增益系数与卷积效应解耦，信道系数将作为后处理与结果Hs相除
        kernel[0:MM] = 1.
        output_pad = wiener_deconv(samples.flatten(),kernel,args.EsN0dB)
        x_re_mat = output_pad[:(LL*MM)].reshape(LL,MM).T  # 这里输出的是Hs, 即每个信号还保留了信道增益
        # print(x_re_mat.flatten()[:5])   # debug
        hh_expand = np.tile(hh,(1,x_re_mat.shape[1]))
        x_re_mat = x_re_mat/hh_expand
        # print(x_re_mat)   # debug
        output = np.sum(x_re_mat,axis=0)
        # return output
        outputs[4] = output.copy()
    
    # Wiener-Denoise estimator
    # if args.Estimator == 5:
        pad_len = len(samples)
        # print(samples)  # debug
        # print(MM,hh.shape)    # debug
        output,_ = inference(samples.flatten(),'../model_new.pth',MM,LL,hh,args.EsN0dB)
        # return output
        outputs[5] = output.copy()

        return outputs

def wiener_deconv(input, kernel, SNR_db=0):       
    ''' 
    Winner filter with given blur kernel
    ---
    input: M长序列，可能包含周边padding
    kernel: 补全至M的卷积核
    '''
    K = 1 / (np.power(10, SNR_db / 10))
    input_fft = fft.fft(input)
    kernel_fft = fft.fft(kernel)
    kernel_fft_1 = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
    result = fft.ifft(input_fft * kernel_fft_1)
    # result = np.abs(fft.fftshift(result))
    return result

def pre_inverse_kernel(kernel):
    kernel = kernel.flatten()
    kernel_new = np.flip(kernel)
    kernel_new = np.roll(kernel_new,1)
    return kernel_new

def BP_Decoding(samples, M, L, hh, noisePowerVec):
    # Prepare the Gaussian messages (Eta,LambdaMat) obtained from the observation nodes
    # Lambda
    beta1 = np.c_[np.real(hh),np.imag(hh)]
    beta2 = np.c_[-np.imag(hh),np.real(hh)]
    Obser_Lamb_first = np.c_[np.matmul(beta1,np.transpose(beta1)),np.matmul(beta1,np.transpose(beta2))]
    Obser_Lamb_second = np.c_[np.matmul(beta2,np.transpose(beta1)),np.matmul(beta1,np.transpose(beta1))]
    Obser_Lamb = np.r_[Obser_Lamb_first,Obser_Lamb_second]
    def fill_mat_with_piece(piece,obj_len):
        '''Fill a (obj_len x obj_len) matrix B, with repeated pieces of A'''
        num_tile = (obj_len // piece.shape[0]) + 1
        out = np.tile(piece,(num_tile,num_tile))
        out = out[0:obj_len,0:obj_len]
        return out
    element = np.zeros([4,4])
    element[0,0] = 1
    ObserMat1 = fill_mat_with_piece(element,Obser_Lamb.shape[0]) * Obser_Lamb # pos-by-pos multiplication
    element = np.zeros([4,4])
    element[0:2,0:2] = 1
    ObserMat2 = fill_mat_with_piece(element,Obser_Lamb.shape[0]) * Obser_Lamb # pos-by-pos multiplication
    element = np.zeros([4,4])
    element[0:3,0:3] = 1
    ObserMat3 = fill_mat_with_piece(element,Obser_Lamb.shape[0]) * Obser_Lamb # pos-by-pos multiplication
    element = np.ones([4,4])
    ObserMat4 = fill_mat_with_piece(element,Obser_Lamb.shape[0]) * Obser_Lamb # pos-by-pos multiplication
    element = np.zeros([4,4])
    element[1:,1:] = 1
    ObserMat5 = fill_mat_with_piece(element,Obser_Lamb.shape[0]) * Obser_Lamb # pos-by-pos multiplication
    element = np.zeros([4,4])
    element[2:,2:] = 1
    ObserMat6 = fill_mat_with_piece(element,Obser_Lamb.shape[0]) * Obser_Lamb # pos-by-pos multiplication
    element = np.zeros([4,4])
    element[3:,3:] = 1
    ObserMat7 = fill_mat_with_piece(element,Obser_Lamb.shape[0]) * Obser_Lamb # pos-by-pos multiplication
    # Eta = LambdaMat * mean
    etaMat = np.matmul(np.r_[beta1,beta2],np.r_[np.real([samples]),np.imag([samples])])

    # process the boundaries
    etaMat[[1,2,3,5,6,7],0] = 0
    etaMat[[2,3,6,7],1] = 0
    etaMat[[3,7],2] = 0
    etaMat[[0,4],-3] = 0
    etaMat[[0,1,4,5],-2] = 0
    etaMat[[0,1,2,4,5,6],-1] = 0

    # ============================================================
    # ============================================================
    # ============================================================ right message passing
    R_m3_eta = np.zeros([2*M, M*(L+1)-2])
    R_m3_Lamb = np.zeros([2*M, 2*M, M*(L+1)-2])
    for idx in range(M*(L+1)-2):
        # ----------------------------- message m1(eta,Lamb) from bottom
        m1_eta = etaMat[:,idx] / noisePowerVec[np.mod(idx,M)]
        if idx == 0: # first boundary -- will only be used in the right passing
            ObserMat = ObserMat1
        elif idx == 1: # second boundary
            ObserMat = ObserMat2
        elif idx == 2:# third boundary
            ObserMat = ObserMat3
        elif idx == M*(L+1)-4: # second last boundary
            ObserMat = ObserMat5
        elif idx == M*(L+1)-3: # second last boundary
            ObserMat = ObserMat6
        elif idx == M*(L+1)-2: # last boundary -- will only be used in the left passing
            ObserMat = ObserMat7
        else:
            ObserMat = ObserMat4
        m1_Lamb = ObserMat / noisePowerVec[np.mod(idx,M)]

        # ----------------------------- message m2: right message => product of bottom and left
        if idx == 0: # first boundary
            m2_eta = m1_eta
            m2_Lamb = m1_Lamb
        else:
            m2_eta = m1_eta + R_m3_eta[:,idx-1]
            m2_Lamb = m1_Lamb + R_m3_Lamb[:,:,idx-1]

        # ----------------------------- message m3: sum
        m2_Sigma = np.linalg.pinv(m2_Lamb) # find the matrix Sigma of m2
        pos = [np.mod(idx+1,M), np.mod(idx+1,M)+M] # pos of two variables (real and imag) to be integrated
        # convert m2_eta back to m2_mean to delete columns -> convert back and add zero columns -> get the new m3_eta
        m2_mean = np.matmul(m2_Sigma, m2_eta) # m2_mean
        m2_mean[pos] = 0 # set to zero and convert back to eta (see below)
        m2_Sigma[pos,:] = 0 # delete the rows and columns of m2_Sigma
        m2_Sigma[:,pos] = 0
        m3_Lamb = np.linalg.pinv(m2_Sigma)
        m3_eta = np.matmul(m3_Lamb, m2_mean)
        # ----------------------------- store m3
        R_m3_eta[:,idx] = m3_eta
        R_m3_Lamb[:,:,idx] = m3_Lamb

    # ============================================================
    # ============================================================
    # ============================================================ left message passing
    L_m3_eta = np.zeros([2*M, M*(L+1)-1])
    L_m3_Lamb = np.zeros([2*M, 2*M, M*(L+1)-1])

    for idx in np.arange(M*(L+1)-2, 0, -1):
        # ----------------------------- message m1: from bottom
        m1_eta = etaMat[:,idx] / noisePowerVec[np.mod(idx,M)];
        if idx == 0: # first boundary -- will only be used in the right passing
            ObserMat = ObserMat1
        elif idx == 1: # second boundary
            ObserMat = ObserMat2
        elif idx == 2: # third boundary
            ObserMat = ObserMat3
        elif idx == M*(L+1)-4: # second last boundary
            ObserMat = ObserMat5
        elif idx == M*(L+1)-3: # second last boundary
            ObserMat = ObserMat6
        elif idx == M*(L+1)-2: # last boundary -- will only be used in the left passing
            ObserMat = ObserMat7
        else:
            ObserMat = ObserMat4

        m1_Lamb = ObserMat / noisePowerVec[np.mod(idx,M)]

        # ----------------------------- message m2: product
        if idx == M*(L+1)-2: # last boundary
            m2_eta = m1_eta
            m2_Lamb = m1_Lamb
        else:
            m2_eta = m1_eta + L_m3_eta[:,idx+1]
            m2_Lamb = m1_Lamb + L_m3_Lamb[:,:,idx+1]

        # ----------------------------- message m3: sum
        m2_Sigma = np.linalg.pinv(m2_Lamb) # find the matrix Sigma of m2
        pos = [np.mod(idx,M), np.mod(idx,M)+M] # pos of two variables (real and imag) to be integrated
        # convert m2_eta back to m2_mean to delete columns -> convert back and add zero columns -> get the new m3_eta
        m2_mean = np.matmul(m2_Sigma, m2_eta) # m2_mean
        m2_mean[pos] = 0 # set to zero and convert back to eta (see below)
        # convert m2_Lambda back to m2_Sigma to delete rows/columns -> convert back and add zero rows/columns -> get the new m3_Lambda
        m2_Sigma[pos,:] = 0
        m2_Sigma[:,pos] = 0
        m3_Lamb = np.linalg.pinv(m2_Sigma)
        m3_eta = np.matmul(m3_Lamb, m2_mean)
        # ----------------------------- store m3
        L_m3_eta[:,idx] = m3_eta
        L_m3_Lamb[:,:,idx] = m3_Lamb

    # ------------------------- Marginalization & BP DECODING
    Sum_mu = np.zeros(L) + 1j * 0
    for ii in range(1, L+1):
        idx = ii * M - 1
        
        Res_Eta = etaMat[:, idx] / noisePowerVec[np.mod(idx,M)] + R_m3_eta[:,idx-1] + L_m3_eta[:,idx+1]
        Res_Lamb = ObserMat4 / noisePowerVec[np.mod(idx,M)] + R_m3_Lamb[:,:,idx-1] + L_m3_Lamb[:,:,idx+1]
        # Res_Eta = etaMat[:, idx] / noisePowerVec[np.mod(idx,M)]
        # Res_Lamb = ObserMat4 / noisePowerVec[np.mod(idx,M)]

        # compute (mu,Sigma) for a variable node
        Res_Sigma = np.linalg.pinv(Res_Lamb)
        Res_mu = np.matmul(Res_Sigma, Res_Eta)

        # compute (mu,Sigma) for the sum
        Sum_mu[ii-1] = np.sum(Res_mu[0:M]) + 1j *np.sum(Res_mu[M:])
    
    return Sum_mu

def test():
    from PIL import Image
    from options import args_parser
    np.set_printoptions(precision=1)
    MM = 20
    LL = 256
    args = args_parser()
    args.EsN0dB = 0
    args.phaseOffset = 3
    # # Generate TransmittedSymbols
    # for m in range(MM):
    #     symbols = 2 * np.random.randint(2, size=(2,LL)) - 1
    #     ComplexSymbols = symbols[0,:] + symbols[1,:] * 0j   # only real part get modulation
    #     if m == 0:
    #         TransmittedSymbols = np.array([ComplexSymbols])
    #     else:
    #         TransmittedSymbols = np.r_[TransmittedSymbols, np.array([ComplexSymbols])]

    loaded_image = Image.open('d_data.png').convert('L')
    loaded_array = np.array(loaded_image)
    d_data = (loaded_array.astype('float32') - 128) / 20.  # 将0-255的整数转换回小数
    TransmittedSymbols = d_data + 0j
    # print('ori symbols:',TransmittedSymbols[:,0]) # debug
    target = np.sum(TransmittedSymbols, 0)
    args.Estimator = -1 # test all
    output = per_pkt_transmission(args, MM, TransmittedSymbols.copy())
    # MSE of the aligned_sample estimator
    MSE1 = np.mean(np.power(np.abs(output[0].flatten() - target.flatten()),2))
    print('MSE1 = ', MSE1)

    # MSE of the ML estimator
    MSE2 = np.mean(np.power(np.abs(output[1].flatten() - target.flatten()),2))
    print('MSE2 = ', MSE2)

    # MSE of the ML estimator (My design)
    MSE2 = np.mean(np.power(np.abs(output[2].flatten() - target.flatten()),2))
    print('MSE2\' = ', MSE2)

    # MSE of the SP-ML estimator
    MSE3 = np.mean(np.power(np.abs(output[3].flatten() - target.flatten()),2))
    print('MSE3 = ', MSE3)

    # MSE of the Wiener estimator
    args.Estimator = 4
    MSE4 = np.mean(np.power(np.abs(output[4].flatten() - target.flatten()),2))
    print('MSE4 = ', MSE4)

    # MSE of the Wiener-Denoise estimator
    args.Estimator = 5
    MSE5 = np.mean(np.power(np.abs(output[5].flatten() - target.flatten()),2))
    print('MSE5 = ', MSE5)

if __name__ == "__main__":
    test()

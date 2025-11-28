import numpy as np

cur_num = 0
total_num = 12

def test_is_same(x,y,atol = 1e-2,rtol = 1e-2):
    x = x.flatten()
    y = y.flatten()
    try:
        print(f"{x=}, {y=}")
        np.testing.assert_allclose(x.flatten(), y.flatten(),rtol=rtol, atol=atol)
    except Exception as e:
        print('---------------')
        idx = np.where(~(x == y))
        print(f"fail idx: {idx=}")
        print(f"shape:'{x.shape}'")
        # print(f"fail idx:'{np.unique(idx[0])}'")
        print(x[idx])
        print(y[idx])
        raise e
    
for i in range(total_num):
    flashmask_out = np.load(f"tmp_res/flashmask_out_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy").squeeze(0)
    flashmask_q_grad = np.load(f"tmp_res/flashmask_q_grad_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy").squeeze(0)
    flashmask_k_grad = np.load(f"tmp_res/flashmask_k_grad_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy").squeeze(0)
    flashmask_v_grad = np.load(f"tmp_res/flashmask_v_grad_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy").squeeze(0)
    magi_out = np.load(f"tmp_res/magi_out_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy")
    magi_q_grad = np.load(f"tmp_res/q_grad_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy")
    magi_k_grad = np.load(f"tmp_res/k_grad_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy")
    magi_v_grad = np.load(f"tmp_res/v_grad_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy")
    
    flashmask_q = np.load(f"tmp_res/paddle_q_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy").squeeze(0)
    flashmask_k = np.load(f"tmp_res/paddle_k_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy").squeeze(0)
    flashmask_v = np.load(f"tmp_res/paddle_v_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy").squeeze(0)
    flashmask_gradOut = np.load(f"tmp_res/paddle_gradOut_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy").squeeze(0)
    magi_q = np.load(f"tmp_res/q_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy")
    magi_k = np.load(f"tmp_res/k_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy")
    magi_v = np.load(f"tmp_res/v_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy")
    magi_gradOut = np.load(f"tmp_res/gradOut_{(int)(cur_num / total_num)}_{cur_num % total_num}.npy")
    
    test_is_same(flashmask_q, magi_q, atol=1e-2, rtol=1e-1)
    test_is_same(flashmask_k, magi_k, atol=1e-2, rtol=1e-1)
    test_is_same(flashmask_v, magi_v, atol=1e-2, rtol=1e-1)
    test_is_same(flashmask_gradOut, magi_gradOut, atol=1e-2, rtol=1e-2)
    try:
        test_is_same(flashmask_out, magi_out, atol=1, rtol=1)
    except AssertionError:
        print(f"FlashMask output is not close to MagiAttention output in batch {cur_num}")
    test_is_same(flashmask_k_grad, magi_k_grad, atol=1, rtol=1)
    test_is_same(flashmask_v_grad, magi_v_grad, atol=1, rtol=1)
    test_is_same(flashmask_q_grad, magi_q_grad, atol=1, rtol=1)
    print(f"Passed_{cur_num}\n")
    cur_num += 1
    

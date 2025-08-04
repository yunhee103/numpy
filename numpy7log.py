# 로그 변환 : 편차가 큰 데이터를 로그변환하면 분포를 개선하고, 큰 범위 차이를 줄임. 
# 모델이 보다 안정적으로 학습할 수 있도록 만들어 주는 장점
import numpy as np
np.set_printoptions(suppress=True, precision=6) #소수 6번째자리
def test():
    values = np.array([3.45, 34.5, 0.345, 0.01, 0.1, 10, 100, 1000])
    print(np.log2(3.45), np.log10(3.45), np.log(3.45))
    print('원본 자료 : ', values)
    log_values = np.log10(values) #상용로그 
    print('log_values : ', log_values)
    ln_values = np.log10(values) #자연로그 
    print('log_values : ', log_values)

    # 로그값의 최소, 최대를 구해 0~1 사이 범위로 정규화
    # 2가지 방법 (표준화 / 정규화)
    # 표준화 : 값을 평균을 기준으로 분포시킴  = 변수값 - 평균 /표준편차
    # 정규화 : 정규화는 데이터 범위를 0~1 사이로 변환해 데이터 분포를 조정 =  변수값 - 최솟값 / 최대값 - 최솟값 
    min_log = np.min(log_values)
    max_log = np.max(log_values)
    normalized = (log_values - min_log) / (max_log - min_log)
    print('정규화 결과 : ', normalized)

def log_inverse():
    offset = 1
    log_values = np.log(10 + offset) # 로그 변환
    print('로그 변환된 값 : ', log_values)
    original= np.exp(log_values) - offset # np.exp() 로그 변환에 역변환 가능
    print('역변환된 값 : ', original)

class LogTrans:
    def __init__(self, offset: float = 1.0):
        self.offset = offset
    
    # 로그 변환을 수행하는 메서드
    def transform(self, x:np.ndarray):
        # fx() = log(x + offset)        # offset을 더하는 이유는 x가 0 이하일 때 로그 계산이 불가능하기 때문
        return np.log(x + self.offset) # 자연로그를 취함 (스케일이 줄어듬)
    
    # 역변환을 수행하는 메서드
    def inverse_trans(self, x_log:np.ndarray):
        return np.exp(x_log) - self.offset # np.exp()를 사용하여 역변환
    


    def normalize(self):
        return (self.log_values - self.min_log) / (self.max_log - self.min_log)
 
print('~' * 20)

def gogo():
    data = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], dtype=float)
    # 로그 변환용 클래스 개체 생성
    log_trans = LogTrans(offset=1.0)
    # 로그 변환 및 역변환
    data_log_scaled = log_trans.transform(data)
    recover_data = log_trans.inverse_trans(data_log_scaled)
    print('원본 데이터 : ', data)
    print('로그 변환된 데이터 : ', data_log_scaled)
    print('역변환된 데이터 : ', recover_data)

if __name__ == "__main__" :
    test()
    log_inverse()
    gogo()  # 함수 호출

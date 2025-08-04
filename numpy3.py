# 배열 연산
import numpy as np  

x = np.array([[1,2],[3,4]], dtype=np.float64)  # 8byte 실수형 배열
y = np.arange(5, 9).reshape(2,2)
y = y.astype(np.float64)  # 8byte 실수형 배열로 변환
print(x, x.astype, x.dtype)
print(y, y.astype, y.dtype)

# 요소별 합 
print(x + y)  # 파이썬 제공 산술연산자
print(np.add(x, y))  # numpy add 함수  
# np.subtract(x, y) , np.multiply, np.divide 제공
import time
big_arr = np.random.rand(1000000) # 100만개의 난수 생성
start = time.time()
sum(big_arr)
end = time.time()
print(f"sum():{end - start:.6f}sec")
start = time.time()
np.sum(big_arr) 
end = time.time()
print(f"np.sum():{end - start:.6f}sec")   
# numpy가 훨씬 속도가 빠르기에 써야함

# 요소별 곱 
# 머신러닝을 쓰기 위한 배움
print(x)
print(y)
print(x * y)  # 파이썬 제공 산술연산자
print(np.multiply(x, y))  # numpy multiply 함수

print(x.dot(y)) # 내적 연산
print()
v = np.array([9,10])
w = np.array([11,12])
print(v * w) # 요소별 곱
print(v.dot(w)) # 내적 연산
print(np.dot(v, w)) # numpy dot 함수
print(np.dot(w, v)) # 행렬곱 연산

print(np.dot(x, y)) # 행렬과 벡터의 곱

print('유용 함수-------------------')
print(x)
print(np.sum(x)) 
print(np.sum(x, axis=0))  # 열 단위 연산 (칼람)
print(np.sum(x, axis=1))  # 행 단위 연산

print(np.min(x), '', np.max(x))  # 최소값, 최대값
print(np.argmin(x), '', np.argmax(x))  # 최소값, 최대값의 인덱스 반환
print(np.cumsum(x))  # 누적합
print(np.cumprod(x))  # 누적곱
print(np.mean(x))  # 평균
print(np.std(x))  # 표준편차    

print()
names = np.array(['tom', 'james', 'oscar', 'tom', 'oscar', 'abc'])
names2 = np.array(['tom', 'page', 'john'])
print(np.unique(names))  # 중복 제거
print(np.intersect1d(names, names2))  # 교집합      
print(np.intersect1d(names, names2, assume_unique=True))  # 중복 제거 후 교집합
print(np.union1d(names, names2))  # 합집합
# help(np.unique) # unique 함수 도움말

print('전치(transpose)')
print(x)
print(x.T)  # 전치 행렬 
arr = np.arange(1,16).reshape(3,5) #3행5열 행렬
print(arr)
print(arr.T)  # 전치 행렬   
print(np.dot(arr.T, arr))  # 내적 연산 가능
# 차원 축소
print(arr.flatten())  # 1차원 배열로 변환
print(arr.ravel())  # 1차원 배열로 변환 (flatten과 동일) 

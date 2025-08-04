# Broadcasting : 크기가 다른 배열 간의 연산시 배열의 구조 자종 변환
# 작은 배열과 큰배열 연산시 작은 배열은 큰배열에 구조를 따름

import numpy as np

x = np.arange(1, 10).reshape(3, 3) # 2차원
y = np.array([1, 0, 1]) # 1차원 

print(x)
print(y)

# 두 배열의 요소 더하기
# 1) 새로운 배열을 이용 
z = np.empty_like(x)  # x와 같은 구조의 빈 배열 생성
print(z)
for i in range(3) :
    for i in range(3) :
        z[i] = x[i] + y
print(z)

# 2) tile을 이용
kbs = np.tile(y, (3, 1)) # y를 3행으로 반복하여 새로운 배열 생성
print('kbs :', kbs)  
z = x + kbs
print(z)

# 3) broadcasting을 이용
# 1D + 1D(같은 길이), 1D + 1D(한쪽 길이1), 2D + 1D가능
# 1D + 1D(길이 다르고 )
kbs = x + y
print(kbs)  

a = np.array([0,1,2])
b = np.array([5,5,5])
print(a + b)  # Broadcasting 적용
print(a + 5)  # Broadcasting 적용

print('\n넘파일로 파일 i/o')
np.save('numpy4etc', x) #bianry 형식으로 저장 
np.savetxt('numpy4etc.txt', x) # 텍스트 형식으로 저장
imsi = np.load('numpy4etc.npy') # binary 형식으로 읽기
print(imsi)

mydatas = np.loadtxt('numpy4etc2.txt', delimiter=',') # 텍스트 형식으로 읽기
print(mydatas)
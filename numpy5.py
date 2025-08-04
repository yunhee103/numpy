# 배열에 행 또는 열 추가하기
import numpy as np

aa = np.eye(3) # 단위행렬 3x3  
print('aa :\n', aa)
bb = np.c_[aa, aa[2]]  # aa의 마지막 행을 추가
print(bb)
cc = np.r_[aa, [aa[2]]]  # aa의 마지막 열을 추가
print(cc)

# reshape을 이용한 배열 구조 변경
a = np.array([1,2,3])
print('np.c_:\n', np.c_[a])     # 1차원 배열을 2차원으로 변환
a.reshape(3,1)
print(a)

print('--appnd, insert, deltete--')
#1차원
print(a)
b = np.append(a, [4,5])  # 배열에 요소 추가
print(b)
c = np.insert(a, 2, [6,7])  # [6,7] 추가
print(c)
#d = np.delete(a, 1)  # 인덱스 1의 요소 삭제
#d = np.delete(a, [1])
d = np.delete(a, [1, 2])  # 인덱스 1,2의 요소 삭제  
print(d)
print()
#2차원
aa = np.arange(1, 10).reshape(3, 3)  # 3행3열
print(aa)
print(np.insert(aa, 1, 99)) # 삽입 후 차원 축소
print(np.insert(aa, 1, 99, axis=0))  # 1인 행에 대하여 99로 채움
print(np.insert(aa, 1, 99, axis=1))  # 열기준

print(aa)
bb = np.arange(10, 19).reshape(3,3)
print(bb)
cc = np.append(aa,bb)  # 추가 후 차원 축소
print(cc)
cc = np.append(aa, bb, axis=0)
print(cc)
cc = np.append(aa, bb, axis=1)
print(cc)

print("np.append 연습")
print(np.append(aa, 88))
print(np.append(aa, [[88,88,88]], axis=0))
print(np.append(aa, [[88],[88],[88]], axis=1))

print("delete 연습")
print(np.delete(aa, 1)) # 삭제 후 차원 축소
print(np.delete(aa, 1, axis=0))
print(np.delete(aa, 1, axis=1)) 

# 조건 연산 where(조건 , 참 , 거짓)
x = np.array([1,2,3])
y = np.array([4,5,6])
condData = np.array([True, False, True])
result = np.where(condData, x, y)
print(result)

aa=np.where(x >= 2)
print(aa)
print(x[aa])
print(np.where (x>=2, 'T', 'F'))
print(np.where (x>=2, x, x+100))

bb = np.random.randn(4,4) #정규분표(가우시안분포-중심극한정리)를 따르는 난수
print(bb)
print(np.where(bb>0, 7, bb))
print('배열 결합/분할')
kbs = np.concatenate([x,y]) #배열 결합
print(kbs)

x1, x2 = np.split(kbs,2) #배열 분할
print(x1)
print(x2)
print()
a = np.arange(1,17).reshape(4,4)
print(a)
x1, x2 = np.hsplit(a,2)
print(x1)
print(x2)

x1, x2 = np.vsplit(a,2)
print(x1)
print(x2)

print('복원, 비복원 추출') #구슬 주머니 중 1개 꺼내고 다음것을 다시 넣으면 복원, 안넣으면 비복원
datas = np.array([1,2,3,4,5,6,7])
# 복원 추출
for _ in range(5):
    print(datas[np.random.randint(0, len(datas) -1)], end = ' ')
# 비복원추출 전용 함수 -sample()
print()
import random
print(random.sample(datas.tolist(),5))
print('--------------')
# 추출 함수 : choice() 둘다 사용 가능함
# 복원 추출
print(np.random.choice(range(1, 46), 6))
# 비복원추출
print(np.random.choice(range(1, 46), 6, replace=False))
# 가중치를 부여한 랜덤 추출
ar = 'air book cat d e f god'
ar = ar.split(' ')
print(ar)
print(np.random.choice(ar,3,p=[0.1,0.1,0.1,0.1,0.1,0.1,0.4]))


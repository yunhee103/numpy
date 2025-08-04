# numpy 기본기능
# list는 중복데이터 허용, 온갖데이터 다 넣을 수 있음 / 배열은 type을 통일해야함 int < float < string 
import numpy as np

ss = ['tom', 'james', 'oscar' ,5]
print(ss, type(ss))
ss2 = np.array(ss)
print(ss2, type(ss2))


# 메모리 비교

li = list(range(1,10))
print(li)
print(id(li[0]), ' ', id(li[1]))
print(li*10) 
print('^' *10)
# list comprehension 
for i in li:
    print(i * 10, end=' ')   
print()
print([i*10 for i in li])  # list comprehension 
# numpy array 다르다

print('---')
# numpy array
num_arr = np.array(li)
print(id(num_arr[0]), ' ', id(num_arr[1])) #주소가 같음
print(num_arr * 10) 
# list 는 빅데이터 처리할 때 메모리 효율이 떨어짐(속도느림). tuple 빠름, 중복허용 set, dict 키값으로 빠름  => c언어로 씀     
# numpy array는 c언어로 작성 data 주소 포인터의 역할, 인덱스로 호출,  

print()
a = np.array([1,2,0,3.0])
print(a, type(a), a.dtype, a.shape, a.ndim, a.size) 
print(a[0], a[1])  # 인덱스 호출
b = np.array([[1,2,3],[4,5,6]]) #행렬 2행3열
print(b.shape, ' ', b[0], ' ', b[[0]])
print(b[0,0], '', b[1,2]) 
print()

c = np.zeros((2,2))
print(c)
d = np.ones((2,2))
print(d)
e = np.full((2,2), fill_value=7)
print(e)
f = np.eye(3)  # 단위행렬 3x3
print(f)
print()
print(np.mean(np.random.rand(500)))  # 0~1 사이의 난수 (균등분포)
print(np.mean(np.random.randn(500))) # 표준정규분포 

np.random.seed(42)  # 난수 고정
print(np.random.randn(2,3))  # 2행3열의 표준정규분포 난수 ( 딥러닝 난수 생성에 많이 사용됨 )

print('\n배열 인덱싱 슬라이싱-------------')
a = np.array([1,2,3,4,5])
print(a)
print(a[1])
print(a[1:]) 
print(a[1:5]) 
print(a[1:5:2]) 
print(a[-2:]) 
print()
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a)
print(a[:])    
print(a[1:])        
print(a[1:, 0:2])  # 1행부터 시작, 0열부터 2열까지)
print(a[0], '', a[0][0], '', a[[0]]) # 0행 전체, 0행의 0열, 0행을 리스트로 

print()
aa=np.array((1,2,3))
print(aa)
bb = aa[1:3]  # subarray 생성(논리적)
print(bb, '', bb[0]) 
bb[0] = 33
print(bb)   
print(aa)
cc = aa[1:3].copy()  # copy()를 사용하여 새로운 배열 생성
print(cc)
cc[0] = 55      
print(cc)
print(aa)  # aa는 변경되지 않음

print('-----------------------')
a = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 3행3열
r1 = a[1, :]    # 1차원
r2 = a[1:2, :]  # 1행 전체 (2차원 배열로 반환)
print(r1, r1.shape)    #    (3,) 1차원 배열
print(r2, r2.shape)    #    (1, 3) 2차원 배열

c1 = a[:, 1]   # 1열 전체 (1차원 배열)
c2 = a[:, 1:2]  # 1열부터 2열까지 (2차원 배열)
print(c1, c1.shape)    #    (3,) 1차원 배열 
print(c2, c2.shape)    #    (3, 1) 2차원 배열

print()
print(a)
bool_idx = (a >= 5)  # 불리언 인덱싱
print(bool_idx)  # 조건에 맞는 위치에 True, 나머지는 False
print(a[bool_idx])  # 조건에 맞는 값들만 출력

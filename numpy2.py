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

for i in li:
    print(i * 10, end=' ')   
print()
print([i*10 for i in li])  # list comprehension 
# numpy array 다르다

print('---')

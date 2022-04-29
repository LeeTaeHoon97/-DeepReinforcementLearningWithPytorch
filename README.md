# -DeepReinforcementLearningWithPytorch

Convert keras to pytorch for Personal Study

Inspired :https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188
## game
#### connect 4
![Connect_Four](https://user-images.githubusercontent.com/59239082/163004555-438f6f0e-2079-4b39-8c35-307b79459665.gif)



## model layers
![keras_model](https://user-images.githubusercontent.com/59239082/162997653-c3a17ddc-1c2a-4c9f-9ca1-118de9ef40ad.png)


## loss graph
![output](https://user-images.githubusercontent.com/59239082/162997545-770eed45-c867-4271-b576-8242bb333c34.png)


## result

#### 학습초기
![before learning](https://user-images.githubusercontent.com/59239082/162998613-7d489ab2-7c3b-43fa-a0d0-beab7ad0525d.jpg)

#### 약 40시간 학습 뒤
![after learning](https://user-images.githubusercontent.com/59239082/162998683-c8a205e4-38d8-4980-9437-72e46611aeb3.jpg)

중앙을 먼저 선점하는쪽이 유리하다는것이 학습됨.

## 연습경기
#### player1 = 초기버전 , player2 = 최신버전
![image](https://user-images.githubusercontent.com/59239082/162999860-3dc1a852-478e-4691-a493-3c77e0abacf0.png)
###### 최신버전이 초기버전보다 승률이 높음.

###### 학습이 좀더 가능할것으로 보이나, 유지자원 및 환경 여건의 문제로 학습종료.

## 막혔던점
### 케라스에서 사용한 함수와 대응되는 파이토치 함수를 찾는것이 힘들었음. 
##### 직접 찾아서 대체
### 모델 레이어 끝단에서 레이어가 2가지로 나뉘는 경우에 대해서는 이번이 처음이라, cost값들을 어떻게 처리해야되는지 어려웠음.
##### cost를 더하는것으로 해결된다는걸 확인
### 처음 학습할 경우 학습 중간에 에러가 발생하는 경우가 발생
##### 어디서 정확이 문제가 발생하는지 몰라, cost값들을 출력해봤는데, ph값들이 점점 작아지다가 nan값이 되는것 발견
##### gradient vanishing 문제로 보임. 일반적인 cnn은 dropout등을 사용하는거로 알았으나 자가학습의 경우 어떻게 처리해야 될지 막막했음  
##### 구글링 결과 nan값들을 0으로 대체하는 코드를 넣음으로써 nan값들을 제거하는 경우가 있다고 확인.
##### 이후 코드는 정상적으로 진행되었으나 loss함수가 뒤죽박죽 되어서 출력(loss가 음수로 내려갈수있는데 nan값이 출력될 경우 0으로 바뀌면서 뒤죽박죽 된거로 보임)
##### 추가적으로 forward 시 발생하는 문제인지, backward시 발생하는 문제인지 알아본 결과 backward문제인거로 확인.
##### 원인은 케라스에서 자체구현하여 사용하였던 loss function이 문제였음. 이를 파이토치 loss function으로 변경하니 정상적으로 학습 시작


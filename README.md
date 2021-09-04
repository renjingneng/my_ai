### 框架结构
```text
root                           
├─ example    // 示例                
│  └─ model.py    // 经典神经网络示例                     
│  └─ transform.py    // pytorch的transform示例                        
├─ lib    // 类库     
│  └─ animator.py  // 动态展示数据，工厂设计模式                         
│  └─ load.py  // 加载标准dataset                          
│  └─ model.py  // 模型用的                         
│  └─ summary.py  // 命令行展示总结                        
│  └─ plt.py  // matplotlib展示 
├─ resource  // 数据资源     
│  └─ dataset  // dataset数据资源     
│  │  ├─ img  //图片             
│  │  └─ text  //文字
│  └─ misc  // 其他                          
│  └─ model  // 训练的模型                    
├─ main.py  
├─ .gitignore
├─ README.md
└─ utility.py // 工具方法   
```   
### 框架依赖
``` 
静态可视化-matplotlib
动态可视化-visdom
数据操作-numpy,pandas 
数学计算-SciPy
传统机器学习-scikit-learn
神经网络框架-pytorch,torchvision
其他-torchinfo
语言处理-nltk
```
### 代码风格
``` 
引入本地的module都是import package/module方式(除了__init__.py文件)，避免污染命名空间
类大写，变量名和方法下划线方式
文件夹名字用一个单词
文件名字用下划线方式
package内部的文件不存在相互引用，如有必要则放在__init__.py文件里面
```
### 计划
- [X] 框架基本结构
- [X] visdom展示训练过程
- [ ] 训练过程中可以保存
- [X] 训练都是在gpu中进行
- [X] LeNet改良版
- [ ] RNN
- [ ] transformer
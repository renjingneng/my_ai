### 框架结构
```text
root                           
├─ example    // 示例                
│  └─ train.py    // pytorch的train示例                        
│  └─ transform.py    // pytorch的transform示例                        
├─ lib    // 类库     
│  └─ load.py  // 加载标准dataset                          
│  └─ model.py  // 模型用的                         
│  └─ show_cmd.py  // 命令行展示数据                          
│  └─ show_plt.py  // 图表展示数据  
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
- [ ] visdom展示训练过程
- [ ] 训练过程中可以保存
- [ ] 训练都是在gpu中进行
- [ ] LeNet改良版
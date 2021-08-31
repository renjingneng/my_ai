### 框架结构
```text
root                           
├─ example    // 示例                
│  └─ transform.py    // pytorch的transform示例                        
├─ lib    // 类库     
│  └─ load.py  // 加载标准dataset                          
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
└─ README.md
```   
### 框架依赖
``` 
静态可视化-matplotlib
数据操作-numpy,pandas 
数学计算-SciPy
传统机器学习-scikit-learn
神经网络框架-pytorch,torchvision
其他-torchinfo
```
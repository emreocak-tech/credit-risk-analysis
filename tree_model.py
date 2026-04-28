import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,train_test_split
from abc import ABC,abstractmethod
from dotenv import load_dotenv
load_dotenv()
import os
file_path=os.getenv("FILE_PATH")
df=pd.read_csv(file_path)

print(df.columns)
print(df['person_age'].unique())
print(df['person_income'].unique())
print(df['person_emp_length'].unique())
print(df['loan_amnt'].unique())
print(df['loan_int_rate'].unique())
print(df['loan_status'].unique())
print(df['loan_percent_income'])
print(df['cb_person_default_on_file'].unique())
print(df['cb_person_cred_hist_length'].unique())
print(df['loan_grade'].unique())
print(df['loan_intent'].unique())
print(df['person_home_ownership'].unique())


df=df.dropna()

categorical_string_values=['person_home_ownership','loan_intent','loan_grade']
object_columns = df.select_dtypes(include=['object']).columns
df=pd.get_dummies(df,columns=categorical_string_values,drop_first=True)
remaining_objects = df.select_dtypes(include=['object']).columns
if len(remaining_objects) > 0:
    print(f"UYARI: Hala string sütunlar var: {list(remaining_objects)}")
    df = pd.get_dummies(df, columns=remaining_objects, drop_first=True)
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"Hata: {col} sütunu hala object tipinde!")
        df[col] = pd.to_numeric(df[col], errors='coerce')
scaler=StandardScaler()
model=DecisionTreeClassifier(class_weight='balanced',max_depth=3, min_samples_leaf=10)

param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}

x=df.drop('loan_status',axis=1)
y=df['loan_status']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True,stratify=y)

scaler.fit(x_train)

class AbstractDefineXandY(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def return_x_train(self,x_train_get):
        pass
    @abstractmethod
    def return_x_test(self,x_test_get):
        pass
    @abstractmethod
    def return_y_train(self,y_train_get):
        pass
    @abstractmethod
    def return_y_test(self,y_test_get):
        pass

class DefineXandY(AbstractDefineXandY):
    def return_x_train(self,x_train_get):
        return x_train
    def return_x_test(self,x_test_get):
        return x_test
    def return_y_train(self,y_train_get):
        return y_train
    def return_y_test(self,y_test_get):
        return y_test

class AbstractScaleXandY(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def scaled_x_train(self):
        pass
    @abstractmethod
    def scaled_x_test(self):
        pass

class ScaledXandY(AbstractScaleXandY):
    def __init__(self):
        define_class=DefineXandY()
        self.x_train=define_class.return_x_train(x_train)
        self.x_test=define_class.return_x_test(x_test)
    def scaled_x_train(self):
        scaled_x_train=scaler.fit_transform(self.x_train)
        return scaled_x_train
    def scaled_x_test(self):
        scaled_x_test=scaler.transform(self.x_test)
        return scaled_x_test

class AbstractModel(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def func_model(self,new_customer):
        pass
class Model(AbstractModel):
    def func_model(self,new_customer):
        scaled_values=ScaledXandY()
        y_train_get=DefineXandY()
        x_train_for_model=scaled_values.scaled_x_train()
        y_train_for_model=y_train_get.return_y_train(y_train)
        grid_search=GridSearchCV(estimator=model,param_grid=param_grid,verbose=1,cv=5,n_jobs=-1,scoring='accuracy')
        grid_search.fit(x_train_for_model,y_train_for_model)
        best_model=grid_search.best_estimator_


        new_customer=scaler.transform(new_customer)

        result=best_model.predict(new_customer)

        return [result,best_model]

class AbstractAccuracyScore(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def accuracy_score(self,new_customer):
        pass

class AccuracyScore(AbstractAccuracyScore):
    def accuracy_score(self,new_customer):
        model_for_accuracy_score=Model()
        scaled_x_test=ScaledXandY()
        get_y_test=DefineXandY()
        scaled_x_test_get=scaled_x_test.scaled_x_test()
        y_test_for_accuracy=get_y_test.return_y_test(y_test)
        best_model=model_for_accuracy_score.func_model(new_customer=new_customer)
        y_pred=best_model[1].predict(scaled_x_test_get)

        accuracy_score_value=accuracy_score(y_test_for_accuracy,y_pred)

        return accuracy_score_value

class AbstractShowImage(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def show_tree_image(self,new_customer):
        pass

class ShowImage(AbstractShowImage):
    def show_tree_image(self,new_customer):
        model_define=Model()
        best_model=model_define.func_model(new_customer=new_customer)
        plt.figure(figsize=(20,10))
        plot_tree(best_model[1],feature_names=x.columns,class_names=['Onaylanmadı', 'Onaylandı'],filled=True,fontsize=10,impurity=True,max_depth=2)
        return plt.gcf()

try:
    scaler.mean_
except:
    scaler.fit(x_train)
    print("Scaler tekrar fit edildi")

fitted_scaler = scaler







































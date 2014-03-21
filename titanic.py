from pandas	import read_csv, pivot_table
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation, svm
from sklearn.metrics import 



def FileOpen(f):

	label = LabelEncoder()
	dicts = {}	

	data = read_csv(f)
	data.pivot_table('PassengerId', 'Pclass', 'Survived', 'count').plot(kind='bar', stacked=True)

	#fig, axes = plt.subplots(ncols=2)
	#data.pivot_table('PassengerId', ['SibSp'], 'Survived', 'count').plot(ax=axes[0], title='SibSp')
	#data.pivot_table('PassengerId', ['Parch'], 'Survived', 'count').plot(ax=axes[1], title='Parch')

	cab = data[data.Embarked.isnull()]

	# This expression shows how changes an empty value of Embarked
	MaxPassEmbarked = data.groupby('Embarked').count()['PassengerId']
	data.Embarked[data.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]

	data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

	# Change the value when it is empty
	data.Age[data.Age.isnull()] = data.Age.mean()
	data.Fare[data.Fare.isnull()] = data.Fare.median()

	label.fit(data.Sex.drop_duplicates())
	dicts['Sex'] = list(label.classes_)
	data.Sex = label.transform(data.Sex)

	label.fit(data.Embarked.drop_duplicates())
	dicts['Embarked'] = list(label.classes_)
	data.Embarked = label.transform(data.Embarked)

	return data

data = FileOpen('train.csv')





print(data)
#plt.show()
#print (data)
#print (se)
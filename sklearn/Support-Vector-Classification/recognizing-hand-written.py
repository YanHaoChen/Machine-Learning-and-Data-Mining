import tkinter
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
import numpy as np

digits = datasets.load_digits()

for key, value in digits.items():
	try:
		print (key, value.shape)
	except:
		print (key)

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
	plt.subplot(2, 4, index+  1)
	plt.axis('off')
	plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	plt.title('Training: %i' % label)

#plt.show()


n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))

classifier = svm.SVC(gamma=0.001)

half = round(n_samples/2)

classifier.fit(data[:half], digits.target[:half])

expected = digits.target[half:]

predicted = classifier.predict(data[half:])

print (expected[:10])
print (predicted[:10])

images_and_predictions = list(zip(digits.images[half:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
	plt.subplot(2, 4, index + 5)
	plt.axis('off')
	plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
	plt.title('Prediction: %i' % prediction)

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected,predicted))

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(digits.target_names))
	plt.xticks(tick_marks, digits.target_names, rotation=45)
	plt.yticks(tick_marks, digits.target_names)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

plt.figure()
plot_confusion_matrix(metrics.confusion_matrix(expected,predicted))

print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))

plt.show()

import youtube_dl, os, csv

def get_label(labelName, fileName):
	with open(fileName) as classFile:
		reader = csv.reader(classFile)
		reader.__next__()
		for row in reader:
			if row[2] == labelName:
				return row[1]


def get_data(labelID,path,fileName):
	os.chdir(path)
	with open(fileName) as classFile:
		reader = csv.reader(classFile)
		reader.__next__()
		reader.__next__()
		reader.__next__()
		for row in reader:
			pick = False
			for elem in row:
				if labelID in elem:
					pick = True
			if pick:
				url = 'https://www.youtube.com/watch?v=' + row[0]
				os.system('youtube-dl -k -f mp4 -o temp.%(ext)s ' + url)
				os.system('ffmpeg -i temp.mp4 -ss '+ row[1] + ' -t 10 temp.wav')
				os.system('del temp.mp4')
				os.system('ren temp.wav ' + row[0] + '.wav')



path =r"C:\Users\marti\Documents\NeuralNetworks\myNeuralNetwork"
os.chdir(path)

labelID = get_label('Noise','class_labels_indices.csv')

path =r"C:\Users\marti\Documents\NeuralNetworks\myNeuralNetwork\train_data_noise"
get_data(labelID,path,'balanced_train_segments.csv')
path =r"C:\Users\marti\Documents\NeuralNetworks\myNeuralNetwork\eval_data_noise"
get_data(labelID,path,'eval_segments.csv')


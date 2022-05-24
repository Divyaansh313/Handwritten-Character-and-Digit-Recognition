# Handwritten-Character-and-Digit-Recognition
CNN model for Handwritten Character and Digit Recognition project

## Introduction
In computer technology and artificial intelligence, machine learning and deep learning are 
crucial. Human effort can be lowered in recognising, learning, predicting, and many other areas 
with the application of deep learning and machine learning. 
Handwritten digit and character recognition has grown in popularity, from the ambitious 
beginning of machine learning and deep learning to the professional who has been practicing for 
years. Creating such a system means developing a machine that can read and categorize pictures 
of handwritten digits and characters.
Handwriting recognition technologies are being employed in a variety of applications. 
Handwriting recognition systems, for example, are required for reading and preserving old 
documents, bank cheques, and letters. There are educational programmes that facilitate 
handwriting recognition on electronic devices such as tablets, particularly in the field of 
education. Individuals with physical or mental limitations, as well as youngsters, might benefit 
from tailored apps enabled by handwriting recognition.
The field of text recognition is quite broad. The subareas of handwriting recognition and 
computer writing recognition are subfields of the handwriting recognition domain. The field of 
computer or typewriter recognition can yield quicker and more effective results. Unlike 
handwriting recognition, it is intended to have a better success rate since there are no distinctive 
patterns or boundaries in letters or digits, such as spacing between letters and words.
Handwritten digits from the MNIST database have been famous among the community for many 
decades now, as decreasing the error rate with different classifiers and parameters, as well as 
preprocessing techniques, from 12 percent error rate with linear classifier (1-Layer NN) to 0.23 
percent error rate with hierarchy of 35 convolution neural networks [Yann LeCun, MNIST 
database of handwritten digits]

## Problem Statement
The problem statement is to classify handwritten digits and characters. The goal is to take an image of 
a handwritten digit and character and determine what that digit and character is. The digits range from 
zero (0) through nine (9) and characters from (A-Z) and (a – z).

## Objectives
To develop a machine learning model:
1. To recognize handwritten digit and character. 
2. To use Convolution Neural Network (CNN) to train the model.
3. To train and test the model using MNIST dataset.
4. To determine the accuracy of the model.

## Methodology
1. Import the libraries and load the MNIST dataset
2. Data Pre-process and Normalize
3. Test-Train split
4. Create the model
5. Evaluate the model
6. Output

## Flowchart
![FLOWCHART](https://github.com/Divyaansh313/Handwritten-Character-and-Digit-Recognition/blob/master/Images/Flowchart.png)

## Implementation and Result
### MNIST
![MNIST](https://github.com/Divyaansh313/Handwritten-Character-and-Digit-Recognition/blob/master/Images/Result_MNIST.png)
### EMNIST
![EMNIST](https://github.com/Divyaansh313/Handwritten-Character-and-Digit-Recognition/blob/master/Images/Result_EMNIST.png)
### Model Information
![Model](https://github.com/Divyaansh313/Handwritten-Character-and-Digit-Recognition/blob/master/Images/Model_Info.png)
### Final Output
![Screenshot](https://github.com/Divyaansh313/Handwritten-Character-and-Digit-Recognition/blob/master/Images/Final_Output.png)

## References
1. https://www.researchgate.net/publication/330138223_A_Literature_Review_on_Handwriten_Character_Recognition_based_on_Artificial_Neural_Network [accessed Jan 29 2022].
2. Rajput GG, Anita HB (2012) Handwritten script recognition using DCT, Gabor filter and wavelet features at line level. In: Book title: soft computing techniques in vision science, pp 33– 43. ISBN 978-3-642-25506-9. doi:10.1007/978-3-642-25507-6_4 
3. Bag, S., Bhowmick, P., Harit, G., 2011. Recognition of bengali handwritten characters using skeletal convexity and dynamic programming, in: Emerging Applications of Information Technology (EAIT), 2011 Second International Conference on, pp. 265–268.
4. https://en.wikipedia.org/wiki/Optical_character_recognition#:~:text=History,-See%20also%3A%20Timeline&text=Early%20optical%20character%20recognition%20may,them%20into%20standard%20telegraph%20code.
5. J. Pradeepa, E. Srinivasana, S. Himavathib, "Neural Network Based Recognition System Integrating Feature Extraction and Classification for English Handwritten", International journal of Engineering,Vol.25, No. 2, pp. 99-106, May 2012 
6. Chaudhuri BB, Bera S (2009). Handwritten text line identification in Indian scripts. In: 10th International conference on document analysis and recognition, 2009.ICDAR „09, pp 636–640, 26–29 July 2009. ISBN 978-1-4244-4500-4. INSPEC Accession Number: 10904634
7. A Literature Survey on Handwritten Character Recognition By Ayush Purohit , Shardul Singh Chauhan 
8. Reena Bajaj, Lipika Dey, and S. Chaudhury, “Devnagari numeral recognition by combining decision of multiple connectionist classifiers”, Sadhana, Vol.27, part. 1,pp.-59-72, 2002.
9. https://en.wikipedia.org/wiki/Optical_character_recognition#:~:text=History,-See%20also%3A%20Timeline&text=Early%20optical%20character%20recognition%20may,them%20into%20standard%20telegraph%20code.

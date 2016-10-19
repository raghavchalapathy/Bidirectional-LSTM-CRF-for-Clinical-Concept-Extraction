__author__ = 'raghav'
# Code for ner-bidirectional LSTM -using tensor flow it does not contain CRF the github link is given below
# https://github.com/monikkinom/ner-lstm
# Start by Obtaining basic statistics about data

# Obtain the count of the Notes for training set
i2b2_train_PATH = "/Users/raghav/Documents/Uni/COLING-2016/COLING-2016-dataPreparationCode/i2b2-2010/in-data/train/"
i2b2_test_PATH = "/Users/raghav/Documents/Uni/COLING-2016/COLING-2016-dataPreparationCode/i2b2-2010/in-data/test/"

i2b2_train_TEMP_OUTPATH = "/Users/raghav/Documents/Uni/COLING-2016/COLING-2016-Code/i2b2-2010/out-data/temp/"
i2b2_ignored_SENT_PATH = "/Users/raghav/Documents/Uni/COLING-2016/COLING-2016-Code/i2b2-2010/out-data/temp/ignored_sentences/"
i2b2_train_OUTPATH = "/Users/raghav/Documents/Uni/COLING-2016/COLING-2016-Code/i2b2-2010/out-data/train"
i2b2_valid_OUTPATH = "/Users/raghav/Documents/Uni/COLING-2016/COLING-2016-Code/i2b2-2010/out-data/valid"
i2b2_test_OUTPATH = "/Users/raghav/Documents/Uni/COLING-2016/COLING-2016-Code/i2b2-2010/out-data/test"

i2b2CONLL2003PATH = "/Users/raghav/Documents/Uni/COLING-2016/COLING-2016-Code/i2b2-2010/tagger/data/conll2003/"
# Testing the concept file
i2b2_concept_testPATH = "/Users/raghav/Documents/Uni/COLING-2016/COLING-2016-Code/i2b2-2010/in-data/train/concept-test/018636330_DH.con"

glove_pre_vec_embPATH = "/Users/raghav/Documents/Uni/COLING-2016/COLING-2016-Code/i2b2-2010/tagger/pre_trainedVectorEmbeddings/glove/"
word2vec_pre_vec_embPATH = "/Users/raghav/Documents/Uni/COLING-2016/COLING-2016-Code/i2b2-2010/tagger/pre_trainedVectorEmbeddings/word2vec/"
url= "http://127.0.0.1:5000/word2vec/model?word="

randomSeedFile = "/Users/raghav/Documents/Uni/COLING-2016/COLING-2016-Code/i2b2-2010/randomGloveEmbedding.txt"

gloveModelFile = "/Users/raghav/Downloads/glove.840B.300d.txt"
word2vecBinaryModelFile = '/Users/raghav/Downloads/GoogleNews-vectors-negative300.bin'
# These dicts would contain the line number and the sentences that is further used to split it into train and validation set
train_sent_Dict = {}
train_sentLabels_Dict = {}

# The test dict contain the sentences and the corresponding labels used in evaluation of the model
test_sent_Dict = {}
test_sentLabels_Dict = {}

# These dicts would contain the line number and the sentences that is further used to split it into train and validation set
# Sentence Length to be considered for training and testing
#
SENTENCE_LENGTH = 1
ignored_lineCounterList = []
ignored_train_sent_Dict = {}
ignored_train_sentLabels_Dict = {}

# The test dict contain the sentences and the corresponding labels used in evaluation of the model
ignored_test_sent_Dict = {}
ignored_test_sentLabels_Dict = {}


# obtain the count of the number of entities in training set
import os
import re
import glob
import shutil
import requests
import gensim
import random

def countNumberOfFiles( path ,fileType):
    count = 0
    for c in glob.iglob(path+'/*.'+fileType):
        count+=1

    # print "The number of txt files in the directory is(Glob)", count
    return count



def countEntities( path,wordlist):

    # Open a file
    c_test=0
    c_problem= 0
    c_treatment = 0
    totalentitiesCount = 0

    for filename in glob.iglob(path+'/*.con'):
        # print(filename)
        f = open(filename,"r")
        for line in f:
            # print line
            tag =  str(line).split("||")
            entity =  tag[1].split("=")
            # print entity

            entity[1]= entity[1].replace("\"","")
            entity[1]= entity[1].replace("\n","")
            # print entity[1]

            if str(wordlist[0]) == str(entity[1])  :
                c_test+= 1
            if (wordlist[1] == entity[1])  :
               c_problem+= 1
            if (wordlist[2] == entity[1])  :
               c_treatment+= 1



        # print [c_test ,c_problem , c_treatment,totalentitiesCount]
    return [c_test ,c_problem , c_treatment]

# Code to count the no of training and test files that is to be included in basic statistics

noOftrain_Files = countNumberOfFiles(i2b2_train_PATH+"/txt","txt")
noOftrain_Concepts= countNumberOfFiles(i2b2_train_PATH+"/concept","con")
print "#(Training Files, Concept Files)", noOftrain_Files,noOftrain_Concepts
print " Acutal no of files Reported files are ","349, 349"

noOftest_Files = countNumberOfFiles(i2b2_test_PATH+"/txt","txt")
noOftest_Concepts= countNumberOfFiles(i2b2_test_PATH+"/concept","con")
print "#(Test Files, Concept Files)", noOftest_Files,noOftest_Concepts
print " Acutal no of files Reported files are ","477, 477"

# Code to count the number of entities present in training and test to be reported in basic statistics table
# The code counts the individual entities as well as total
# the types of entities present are problem, treatment, test
wordlist = ["test","problem","treatment"]
[c_test1 ,c_problem1 , c_treatment1] = countEntities(i2b2_train_PATH+"/concept/",wordlist)
totalentitiesCount=  c_test1 + c_problem1 + c_treatment1
print "Training Entities.. [c_test ,c_problem , c_treatment,totalentitiesCount]", [c_test1 ,c_problem1 , c_treatment1,totalentitiesCount]

[c_test2 ,c_problem2 , c_treatment2] = countEntities(i2b2_test_PATH+"/concept",wordlist)
totalentitiesCount=  c_test2 + c_problem2 + c_treatment2
print "Test Entities.. [c_test ,c_problem , c_treatment,totalentitiesCount]", [c_test2 ,c_problem2 , c_treatment2,totalentitiesCount]

# Computed the basic statistics

# Create the CONLL-2003 data format

# Read the data file

def computeConceptDict(conFilePath):
    cf = open(conFilePath,"r")
    cf_Lines = cf.readlines()
    line_dict = dict()


    for cf_line in cf_Lines:
        # print cf_line
        #c="a workup" 27:2 27:3||t="test"
        concept= cf_line.split("||")

        iob_wordIdx = concept[0].split()
        # print concept[0]
        iob_class = concept[1].split("=")
        iob_class = iob_class[1].replace("\"","")
        iob_class = iob_class.replace("\n","")

        # print iob_wordIdx[len(iob_wordIdx)-2],iob_wordIdx[len(iob_wordIdx)-1]
        start_iobLineNo = iob_wordIdx[len(iob_wordIdx)-2].split(":")
        end_iobLineNo = iob_wordIdx[len(iob_wordIdx)-1].split(":")
        start_idx = start_iobLineNo[1]
        end_idx = end_iobLineNo[1]
        iobLineNo=start_iobLineNo[0]
        # print "start",start_idx
        # print "end",end_idx

        # print "line Number, start_idx,end_idx, iobclass",iobLineNo,start_idx,end_idx,iob_class
        # line_dict.update({iobLineNo:start_idx+"-"+end_idx+"-"+iob_class})

        if iobLineNo in line_dict.keys():
                # append the new number to the existing array at this slot
                # print "Found duplicate line number....."
                line_dict[iobLineNo].append(start_idx+"-"+end_idx+"-"+iob_class)
        else:
                # create a new array in this slot
                line_dict.update({iobLineNo:[start_idx+"-"+end_idx+"-"+iob_class]})

    #
    # for k,v in line_dict.iteritems():
    #     print k,v

    return line_dict


def prepareIOB_wordList(wordList,lineNumber,IOBwordList,conceptDict,dataType):
    # print "Line Number",lineNumber
    # print "Word- List ",wordList

    iobTagList= []


    if str(lineNumber) in conceptDict.keys():
         # print conceptDict[str(lineNumber)]

         # split the tag and get the index of word and tag
         for concept in conceptDict[str(lineNumber)]:
             concept = str(concept).split("-")
             # print "start_idx, end_idx",concept[0],concept[1]
             # if (start_idx - end_idx) is zero then only B- prefix is applicable
             getrange = range(int(concept[0]),int(concept[1]))
             getrange.append(int(concept[1]))
             # For all the idx not in getrange assign an O tag
             # print getrange


             if(len(getrange) > 1):

                     for idx in range(0,len(getrange)):
                         # print getrange[idx]
                         iobTagList.append(int(getrange[idx]))
                         if(idx == 0):
                                IOBwordList[getrange[idx]] = "B-"+concept[2]
                         else:
                                 IOBwordList[getrange[idx]] = "I-"+concept[2]

             else:

                     idx = getrange[0]
                     iobTagList.append(int(getrange[0]))
                     # print idx
                     IOBwordList[idx] = "B-"+concept[2]



             # Else for all the indices between start and end apply the I- prefix

            # For all the other words assign O tag
         for i in range(0,len(IOBwordList)):
              if i not in iobTagList:
                IOBwordList[i] = "O"
         # print "IOB- WordList ",IOBwordList
    else:
         # print ""
         for i in range(0,len(IOBwordList)):
              if i not in iobTagList:
                IOBwordList[i] = "O"
         # print "IOB-  List ",IOBwordList
         # print "These Lines have ZERO IOB tags",IOBwordList
         # print "IOB Tag list ",iobTagList




    return IOBwordList




def createCONLL2003Data(inpath, outpath,dataType):

    # Make sure you have deleted all the old output files
    txtPath = inpath+"/txt"
    conPath = inpath+"/concept"

    # remove all the *.txt files present in the path and rewrite it
    removefilesInDirectoryPath(outpath)

    if(dataType == "train"):

        train_Outfile = open(outpath+"/train.txt", "a")
    else:

        test_Outfile = open(outpath+"/test.txt", "a")

    conllfileContent=""
    filecounter = 0
    linecounter = 0
    # get all the list of file names only into the filenames list
    filenamesList = [os.path.basename(x) for x in glob.iglob(txtPath+'/*.txt')]

    for filename in filenamesList:
        filecounter+=1
        # print"The number of files processed are ",filecounter,(filename)
        f = open(txtPath+"/"+filename,'r')
        lines = f.readlines()
        # print "Number of Lines are : " ,len(lines)
        confileName=filename.split(".")
        confileName = confileName[0]
        conceptDict = computeConceptDict(conPath+"/"+confileName+".con")
        # print conceptDict
        # print lines
        for line in range(0 ,len(lines)):
            words =  str(lines[line]).split()
            orginial_wordsList =  str(lines[line]).split()
            linecounter+=1

            IOBwordList= words
            # print words
            lineNumber= line+1 # Line number starts with 1


            #Prepare the IOB word list
            IOBwordList=prepareIOB_wordList(words,lineNumber,IOBwordList,conceptDict,dataType)

            # Merge the words and IOB words list in conll-2003 format

            for w in range(0,len(words)):
                conllfileContent= orginial_wordsList[w] + "\t" + IOBwordList[w] +"\n"
                # print conllfileContent
                if(dataType == "train"):
                    train_Outfile.write(conllfileContent)

                elif(dataType == "test"):
                    test_Outfile.write(conllfileContent)

             # add an Empty Line after each sentence conll 2003 format
            if(dataType == "train"):
                train_Outfile.write("\n")
            else:
                 test_Outfile.write("\n")

            if(dataType == "train"):
                   if(len(orginial_wordsList)== SENTENCE_LENGTH):
                        orginial_wordsList.append("PAD-WORD")
                        IOBwordList.append("O")

                   # print "The total number of sentences added untill now",linecounter,orginial_wordsList,IOBwordList
                   train_sent_Dict.update({linecounter:orginial_wordsList})
                   train_sentLabels_Dict.update({linecounter:IOBwordList})

            else:
                   if(len(orginial_wordsList)== SENTENCE_LENGTH):
                         orginial_wordsList.append("PAD-WORD")
                         IOBwordList.append("O")
                   test_sent_Dict.update({linecounter:orginial_wordsList})
                   test_sentLabels_Dict.update({linecounter:IOBwordList})


    return linecounter

def divideDataIntotrain_valid(inputDict):


    train_count= int(round(0.7 * len(inputDict)))
    valid_count= len(inputDict)- int(round(0.7 * len(inputDict)))
    return [train_count,valid_count]


import random

def createTrain_Valid_Set(maxrange,countTrain,countValid,ignoredLineCounterList):
    #generate a random number between 1 and 6502 and keep adding into list of index of sentences untill train_count
    count = 0
    train_sentList = []
    valid_sentList= []
    from sets import Set
    original_random_train_list = list(Set([]))
    # print "Max Range",maxrange

    for n in range(1,maxrange+1,1):

        original_random_train_list.append(n)

    # print "Original Random List",original_random_train_list
    # print "Ignored sentences are ",ignoredLineCounterList


    # Remove all the ignored keys so that it does not throw  key not found exception in train and validation datasets

    # original_random_train_list = list(set(original_random_train_list)^set(ignoredLineCounterList))

    original_random_train_list= list(set(original_random_train_list).difference(ignoredLineCounterList))

    # print "After Removed Ignored Keys  ",original_random_train_list



    random.shuffle(original_random_train_list)

    # print " After Shuffling the original list",original_random_train_list


    # Split the list into train and valid size
    for e in original_random_train_list:
        if(count <= countTrain):
            train_sentList.append(e)
            count= count +1
        else:
           valid_sentList.append(e)

    # print("Max of random set")
    # print max(original_random_train_list)
    return [train_sentList,valid_sentList]


def removefilesInDirectoryPath(dirPath):
    # Change the directory to the current working directory
    os.chdir(dirPath)
    filelist = [ f for f in os.listdir(dirPath) if f.endswith(".txt") ]
    for f in filelist:
        os.remove(f)

    return



def save_train_validDatasets(train_sent_Dict,train_sentLabels_Dict,i2b2_train_OUTPATH,i2b2_valid_OUTPATH,train_sentList,valid_sentList):

     # First remove all the files in the directory
     removefilesInDirectoryPath(i2b2_train_OUTPATH)
     removefilesInDirectoryPath(i2b2_valid_OUTPATH)
     # Open train and validation file in the specified path
     train_Outfile = open(i2b2_train_OUTPATH+"/train.txt", "a")
     valid_Outfile = open(i2b2_valid_OUTPATH+"/dev.txt", "a")

     for sent_id in train_sentList:
         orginial_wordsList = train_sent_Dict[sent_id]
         IOBwordList = train_sentLabels_Dict[sent_id]
         for w in range(0,len(orginial_wordsList)):
                    conllfileContent= orginial_wordsList[w] + "\t" + IOBwordList[w] +"\n"
                    train_Outfile.write(conllfileContent)
         train_Outfile.write("\n") # after the end of every sentence write an new- line character


     for sent_id in valid_sentList:
         orginial_wordsList = train_sent_Dict[sent_id]
         IOBwordList = train_sentLabels_Dict[sent_id]
         for w in range(0,len(orginial_wordsList)):
                    conllfileContent= orginial_wordsList[w] + "\t" + IOBwordList[w] +"\n"
                    valid_Outfile.write(conllfileContent)
         valid_Outfile.write("\n") # after the end of every sentence write an new- line character # add an Empty Line after each sentence conll 2003 format


     return





def create_validationDataSet(train_sent_Dict,train_sentLabels_Dict,i2b2_train_OUTPATH,i2b2_valid_OUTPATH):

    [countTrain,countValid] = divideDataIntotrain_valid(train_sent_Dict)
    [train_sentList,valid_sentList]=createTrain_Valid_Set(len(train_sent_Dict),countTrain,countValid,ignored_lineCounterList)

    # print "training List of sentences are ",train_sentList
    # print "Validation list of sentences are ",valid_sentList

    print " The number of training sentences are ",countTrain
    print " The number of validation sentences are ",countValid
    print "The total of training and test is ",countTrain+countValid

    save_train_validDatasets(train_sent_Dict,train_sentLabels_Dict,i2b2_train_OUTPATH,i2b2_valid_OUTPATH,train_sentList,valid_sentList)

    return



def generateRandomGloveVector(randomSeedFile):


    randomSeedFile = open(randomSeedFile,"r")

    for line in randomSeedFile:

        line = line.split(" ")
        # print line
        random.shuffle(line,random.random) # shuffle the vector contents randomly
        # print line
        # print len(line)

        result = ' '.join(str(v) for v in line)
        # print result



    return result


def create_testDataSet(test_sent_Dict,test_sentLabels_Dict,i2b2_test_OUTPATH):
    removefilesInDirectoryPath(i2b2_test_OUTPATH)
    # Open train and validation file in the specified path
    test_Outfile = open(i2b2_test_OUTPATH+"/test.txt", "a")
    emptyTestsent=0
    for sent_id in test_sent_Dict:
        orginial_wordsList = test_sent_Dict[sent_id]
        IOBwordList = test_sentLabels_Dict[sent_id]
        # print orginial_wordsList
        # print IOBwordList
        if not orginial_wordsList:
            emptyTestsent+= 1
            # print emptyTestsent
        else:
             for w in range(0,len(orginial_wordsList)):
                conllfileContent= orginial_wordsList[w] + "\t" + IOBwordList[w] +"\n"
                test_Outfile.write(conllfileContent)
             test_Outfile.write("\n") # after the end of every sentence write an new- line character


    return

# computeConceptDict(i2b2_concept_testPATH)


def count_entitiesWithinDict(inputDict):
    c_ig_test =0
    c_ig_problem=0
    c_ig_treatment=0

    # print " Total ignored entities present are: ", len(inputDict)
    print " Total Sentences with one word  entities present are: ", len(inputDict)

    for k,v in inputDict.iteritems():

        v= ''.join(v)


        if(v== "B-test"):
            c_ig_test+= 1
        elif(v == "B-problem"):
            c_ig_problem+=1
        elif(v == "B-treatment"):
            c_ig_treatment+=1



    return [c_ig_test,c_ig_problem,c_ig_treatment]




def loadword2VecModel(word2vecBinaryModelFile):
    print "Loading word2vec Model-!!!!"
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.Word2Vec.load_word2vec_format(word2vecBinaryModelFile, binary=True)

    return model

def loadGloveModel(gloveFile):
    print "Loading Glove Model-!!!!"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print "Done.",len(model)," words loaded-!!!!!"
    return model

def create_pretrained_EmbeddingFile(vocab,pre_vec_embPATH,modelType):
    glovemodel = {}
    word2vecmodel = ""
    pre_embOut=""
    null_embOut=""

    print "Creating pre-trained vector embeddings file...."

    if(modelType == "glove"):
        glovemodel = loadGloveModel(gloveModelFile)
    elif(modelType == "word2vec"):
        word2vecmodel = loadword2VecModel(word2vecBinaryModelFile)

    # pre_embOut = codecs.open(pre_vec_embPATH+"pre_emb.txt", 'a', 'utf-8')

    baseurl = url
    nulcounter = 0
    # recognise if the word in vocab is "number", "date", "alphanumeric" and "others".
    for word in vocab:

        if(modelType == "glove"):
            pre_embOut = open(pre_vec_embPATH+"pre_emb.txt", 'a')
            null_embOut = open(pre_vec_embPATH+"null_pre_emb.txt", 'a')
            try:
                result = glovemodel[word]
                result = ['{:.6f}'.format(x) for x in result]
                result = ' '.join(str(v) for v in result)
            except:
                result = generateRandomGloveVector(randomSeedFile)
                # result = ['{:.6f}'.format(x) for x in result]
                result = ''.join(str(v) for v in result)
                null_embOut.write(word + "\t" + result + "\n" )
            if(result == "null"):
                null_embOut.write(word + "\t" + result + "\n" )



        elif(modelType == "word2vec"):
            pre_embOut = open(pre_vec_embPATH+"pre_emb.txt", 'a')
            null_embOut = open(pre_vec_embPATH+"null_pre_emb.txt", 'a')
            try:
                result = word2vecmodel[word]
                result = ['{:.6f}'.format(x) for x in result]
                result = ' '.join(str(v) for v in result)
                pre_embOut.write(word + "\t" + result + "\n" )
            except:
                result = generateRandomGloveVector(randomSeedFile)
                print "random seed file",result
                # result = ['{:.6f}'.format(x) for x in result]
                result = ''.join(str(v) for v in result)
                print "embeddings....",word + "\t" + result + "\n"
                null_embOut.write(word + "\t" + result + "\n" )
                pre_embOut.write(word + "\t" + result + "\n" )

            if(result == "null" or result == 0.000000 ):
                null_embOut.write(word + "\t" + result + "\n" )



        # pre_embOut.write(word + "\t" + result + "\n" )

    print " total number of words in vocab is ",len(vocab)

    return



#Remove previous train , valid and test files before creating train, test, validation files
train_totalSentences = createCONLL2003Data(i2b2_train_PATH, i2b2_train_TEMP_OUTPATH,"train")
test_totalSentences=createCONLL2003Data(i2b2_test_PATH, i2b2_test_OUTPATH,"test")
print "Total Number of training  sentences",train_totalSentences
print "Total Number of test  sentences",test_totalSentences


# Combine the training and test sentences to create the vocabulary dict
vocab = set()
# append all the words from training sentences
for k, v in train_sent_Dict.iteritems():
    vocab.update( sorted(set(w for w in v)))
# Append all the words from the test sentences
for k, v in test_sent_Dict.iteritems():
    vocab.update( sorted(set(w for w in v)))
print " The Number of words in Vocabulary " ,len(vocab)+1
# For each word in the vocab get a pretrained embedding and store it in pre_emb.txt file
#removefilesInDirectoryPath(glove_pre_vec_embPATH)
#create_pretrained_EmbeddingFile(vocab,glove_pre_vec_embPATH,"glove")

create_pretrained_EmbeddingFile(vocab,word2vec_pre_vec_embPATH,"word2vec")

# Create training and validation data from the training set
create_validationDataSet(train_sent_Dict,train_sentLabels_Dict,i2b2_train_OUTPATH,i2b2_valid_OUTPATH)
# Create the test data set
create_testDataSet(test_sent_Dict,test_sentLabels_Dict,i2b2_test_OUTPATH)




# Copy the final ouput to model training directory
shutil.copy2(i2b2_train_OUTPATH+'/train.txt', i2b2CONLL2003PATH+'/train.txt')
shutil.copy2(i2b2_valid_OUTPATH+'/dev.txt', i2b2CONLL2003PATH+'/dev.txt')
shutil.copy2(i2b2_test_OUTPATH+'/test.txt', i2b2CONLL2003PATH+'/test.txt')




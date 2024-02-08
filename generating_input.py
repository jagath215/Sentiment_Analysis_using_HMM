from Data_processor import Data_processor

my_data = Data_processor("train/test")

outfile = open("Input/input2.in", 'w')
data = my_data.data
# print(data)
for sentence in range(len(data)):
        for word in data[sentence]:
            a, b = word.split(" ")
            output = a + "\n"
            outfile.write(output)
        outfile.write("\n")
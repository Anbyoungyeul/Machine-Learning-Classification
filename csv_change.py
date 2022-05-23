import glob
import csv

some = ["byte", "short", "dword", "ptr"]

def fileter_list(line):
    temp = line.split(" ")
    temp = list(filter(('').__ne__,temp))
    l = len(temp)
    i=0
    while i != (l-1):
        temp[i] = temp[i].strip("\n")
        temp[i] = temp[i].strip(",")
        #merge elements in some
        if temp[i] in some:
            temp[i-1] = temp[i-1] + " " + temp[i]
            temp.remove(temp[i])
            l = l - 1
        else:
            i+=1 
    return temp

if __name__=="__main__":
    Dataset_asm =[]
    for file_num in glob.glob("D:\\disasmbled\\train__benign\\*.asm"):
        Dataset_asm.append(file_num)
    # 어셈 확장자 파일명 획득

    for path in Dataset_asm:
        source = open(path)
        t = path.split('\\')
        t = t[1].split(".")[0]
        
        final = []
        for source_lines in source.readlines():
            source_lines = fileter_list(source_lines)
            final.append(source_lines)

        column_names = ["instruction", "opcode_1", "opcode_2"]
        with open("D:\\csv"+t+'.csv','w+', newline="") as f:
            write = csv.writer(f)
            write.writerow(column_names)
            write.writerows(final)


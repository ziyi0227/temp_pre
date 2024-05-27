import csv


def convert_csv(input_file, output_file):
    # 打开输入的分号分割的CSV文件
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=';')

        # 打开输出的逗号分割的CSV文件
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter=',')

            # 将每行数据从输入文件写到输出文件
            for row in reader:
                writer.writerow(row)


if __name__ == '__main__':
    input_file = 'D:\\大学\\课程\\各课程\\数据科学导论\\期末\\temp_pre\\58457.22.05.2005.22.05.2024.1.0.0.cn.utf8.00000000.csv'
    output_file = 'D:\\大学\\课程\\各课程\\数据科学导论\\期末\\temp_pre\\weather_data.csv'
    convert_csv(input_file, output_file)


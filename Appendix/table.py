import xlsxwriter
def createTable24(K, a, b, c, d):
    workbook = xlsxwriter.Workbook('chart.xlsx')
    worksheet = workbook.add_worksheet()

    # 指定类型为柱状图
    chart = workbook.add_chart({'type': 'column'})

    title = [u'K', u'precesion', u'recall', u'coverage', u'popularity']
    buname= [K]

    data = [
        [a, b, c, d]
    ]

    # 定义format格式对象
    format=workbook.add_format()
    # 定义format对象单元格边框加粗1像素
    format.set_border(1)

    format_title=workbook.add_format()
    format_title.set_border(1)
    # format_title对象单元格背景色为#cccccc
    format_title.set_bg_color('#cccccc')
    # 居中格式
    format_title.set_align('center')
    format_title.set_bold()

    format_ave=workbook.add_format()
    format_ave.set_border(1)
    # 以小数形式显示
    format_ave.set_num_format('0.00')

    # 以行或列的方式写入数据，同时引用格式
    worksheet.write_row('A1',title,format_title)
    worksheet.write_column('A2', buname, format)
    worksheet.write_row('B2', data[0], format)


    workbook.close()


def createTable25(a, b, c, d):
    workbook = xlsxwriter.Workbook('chart.xlsx')
    worksheet = workbook.add_worksheet()

    # 指定类型为柱状图
    chart = workbook.add_chart({'type': 'column'})

    title = [u'Table', u'precesion', u'recall', u'coverage', u'popularity']
    buname= [u'Random']

    data = [
        [a, b, c, d]
    ]

    # 定义format格式对象
    format=workbook.add_format()
    # 定义format对象单元格边框加粗1像素
    format.set_border(1)

    format_title=workbook.add_format()
    format_title.set_border(1)
    # format_title对象单元格背景色为#cccccc
    format_title.set_bg_color('#cccccc')
    # 居中格式
    format_title.set_align('center')
    format_title.set_bold()

    format_ave=workbook.add_format()
    format_ave.set_border(1)
    # 以小数形式显示
    format_ave.set_num_format('0.00')

    # 以行或列的方式写入数据，同时引用格式
    worksheet.write_row('A1',title,format_title)
    worksheet.write_column('A2', buname,format)
    worksheet.write_row('B2', data[0],format)



    # 设置图表上方标题

    workbook.close()
def createTable27(a, b, c):
    workbook = xlsxwriter.Workbook('chart.xlsx')
    worksheet = workbook.add_worksheet()

    # 指定类型为柱状图
    chart = workbook.add_chart({'type': 'column'})

    title = [u'Table', u'movie_id_1', u'movie_id_2', u'similarity']
    buname= [u'1', u'2', u'3']

    data = [
        [a[0], b[0], c[0]],
        [a[1], b[1], c[1]],
        [a[2], b[2], c[2]],
        [a[3], b[3], c[3]],
        [a[4], b[4], c[4]]
    ]
    data1 = [[a[1], b[1], c[1]]]
    print(data)
    # 定义format格式对象
    format=workbook.add_format()
    # 定义format对象单元格边框加粗1像素
    format.set_border(1)

    format_title=workbook.add_format()
    format_title.set_border(1)
    # format_title对象单元格背景色为#cccccc
    format_title.set_bg_color('#cccccc')
    # 居中格式
    format_title.set_align('center')
    format_title.set_bold()

    format_ave=workbook.add_format()
    format_ave.set_border(1)
    # 以小数形式显示
    format_ave.set_num_format('0.00')

    # 以行或列的方式写入数据，同时引用格式
    worksheet.write_row('A1', title, format_title)
    #worksheet.write_column('A2', buname, format)
    worksheet.write_row('B2', data[0], format)

    worksheet.write_row('B2', data1[0], format)
    worksheet.write_row('A2', data1[0], format)
    print(data[1])
    worksheet.write_row('B2', data[0], format)


    workbook.close()


def createTable26(a, b, c, d):
    workbook = xlsxwriter.Workbook('chart.xlsx')
    worksheet = workbook.add_worksheet()

    # 指定类型为柱状图
    chart = workbook.add_chart({'type': 'column'})

    title = [u'Table', u'precesion', u'recall', u'coverage', u'popularity']
    buname= [u'UserCF']

    data = [
        [a, b, c, d]
    ]

    # 定义format格式对象
    format=workbook.add_format()
    # 定义format对象单元格边框加粗1像素
    format.set_border(1)

    format_title=workbook.add_format()
    format_title.set_border(1)
    # format_title对象单元格背景色为#cccccc
    format_title.set_bg_color('#cccccc')
    # 居中格式
    format_title.set_align('center')
    format_title.set_bold()

    format_ave=workbook.add_format()
    format_ave.set_border(1)
    # 以小数形式显示
    format_ave.set_num_format('0.00')

    # 以行或列的方式写入数据，同时引用格式
    worksheet.write_row('A1',title,format_title)
    worksheet.write_column('A2', buname,format)
    worksheet.write_row('B2', data[0],format)


    # 定义图表数据系列函数


    workbook.close()


def getWeekDate(*args):
    year, month, day = args
    year = int(year)
    century = int(year / 100)
    year = year - int(year / 100) * 100
    month = int(month)
    if month == 1 or month == 2:
        month = month + 12
        if year == 0:
            year = 99
            century = century - 1
        else:
            year = year - 1
    day = int(day)
    week = year + int(year / 4) + int(century / 4) - 2 * century + int(26 * (month + 1) / 10) + day - 1
    if week < 0:
        weekDay = (week % 7 + 7) % 7
    else:
        weekDay = week % 7
    return weekDay


if __name__ == '__main__':
    date = input('输入年份月份天数,以空格分割:\t')  # python3 中请使用input('输入年份月份天数,以空格分割:\t')
    year, month, day = date.split(' ')
    print(getWeekDate(year, month, day))

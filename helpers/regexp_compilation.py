import re

# select decimal dot or comma in decimal
save_decimal_fraction_as_dot = re.compile('(\d+)[\.,](\d)+')

# all except letters
remove_set_1 = re.compile("[^[a-zA-Zа-яА-Я]+")

# all except letters and spaces
remove_set_2 = re.compile("[^[a-zA-Zа-яА-Я\s]+")

# all except letters and digits
remove_set_3 = re.compile("[^\w]+")

# all except letters, digits and spaces
remove_set_4 = re.compile("[^\w\s]+")

# all except letters, digits and spaces and '-'
remove_set_5 = re.compile("[^\w\s-]+")

# all except letters, digits and spaces and '-', '?'
remove_set_6 = re.compile("[^\w\s\-\?]+")

# all except letters, digits and spaces and '-', '?', '.',',', '@', '"'
remove_set_7 = re.compile("[^\w\s\?\-@\.\"]+")

# all except letters, digits and spaces and '-', '?', '.', '@', '"', "()", '%'
remove_set_8 = re.compile("[^\w\s\?\-@\.\"()%]+")

# all except exotic
remove_set_9 = re.compile("[^\w\s\?\-@\.,\"()%\+/=;]+")

# all except exotic and quotes
remove_set_10 = re.compile("[^\w\s\?\-@₽\.,$()%\+/=;]+")

# remove multiple spaces
multiple_spaces = re.compile(' +')

# underscores
underscores = re.compile('_+')
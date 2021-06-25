# Subway-Challenge

# Introduction

To have a better understand on how u work, your strengths and weakness, we're proposing a challenge!

You are going to work with the NYC metro database, which has the field description attached below. To help your work, we present the database already merged at this link.


# Field Description

CA,UNIT,SCP,STATION,LINENAME,DIVISION,DATE,TIME,DESC,ENTRIES,EXITS

CA = Control Area (A002) UNIT = Remote Unit for a station (R051) SCP = Subunit Channel Position represents an specific address for a device (02-00-00) STATION = Represents the station name the device is located at LINENAME = Represents all train lines that can be boarded at this station Normally lines are represented by one character. LINENAME 456NQR repersents train server for 4, 5, 6, N, Q, and R trains. DIVISION = Represents the Line originally the station belonged to BMT, IRT, or IND DATE = Represents the date (MM-DD-YY) TIME = Represents the time (hhmmss) for a scheduled audit event DESc = Represent the REGULAR scheduled audit event (Normally occurs every 4 hours) 1. Audits may occur more that 4 hours due to planning, or troubleshooting activities. 2. Additionally, there may be a RECOVR AUD entry This refers to a missed audit that was recovered. ENTRIES = The comulative entry register value for a device EXIST = The cumulative exit register value for a device

Example The data below shows the entryexit register values for one turnstile at control area (A002) from 092714 at 0000 hours to 092914 at 0000 hours

CA,UNIT,SCP,STATION,LINENAME,DIVISION,DATE,TIME,DESC,ENTRIES,EXITS A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-27-14,000000,REGULAR,0004800073,0001629137, A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-27-14,040000,REGULAR,0004800125,0001629149, A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-27-14,080000,REGULAR,0004800146,0001629162, A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-27-14,120000,REGULAR,0004800264,0001629264, A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-27-14,160000,REGULAR,0004800523,0001629328, A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-27-14,200000,REGULAR,0004800924,0001629371, A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-28-14,000000,REGULAR,0004801104,0001629395, A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-28-14,040000,REGULAR,0004801149,0001629402, A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-28-14,080000,REGULAR,0004801168,0001629414, A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-28-14,120000,REGULAR,0004801304,0001629463, A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-28-14,160000,REGULAR,0004801463,0001629521, A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-28-14,200000,REGULAR,0004801737,0001629555, A002,R051,02-00-00,LEXINGTON AVE,456NQR,BMT,09-29-14,000000,REGULAR,0004801836,0001629574,


# Funções Auxiliares utilizadas na modelagem 

Importando funções auxiliares do arquivo fuctions.py para melhor organização e estruturação do codigo

transform_time: Função recebe o dataframe e realiza as transformações temporais adicionando mes, ano, dia e realiza a transformação das datas

parallelize_dataframe: Função de multiprocessamento e parelelização das funções, auxilia na redução do tempo de execução

transform_data: Função recebe o dataframe e realiza tratamentos em algumas colunas e adiciona informações baseadas em datas (feriados, dias da semana, id_mes)

get_encoders: Função que realiza a transformação dos dados utilizando label encoder para transformar dados categoricos em numericos

dummies_test: Função que gera dummy das colunas categoricas pré definidas

lag_features: Função que gera lag (visão passada) das utilizações do metro

roll_mean_features: Função que gera uma janela de media olhando para ranges pre definidos

random_noise: Função que gera um valor aleatorio para servir de ruido para as funções

reduce_mem_usage: Função aplica a redução de memoria do dataframe realizando transformações nas colunas

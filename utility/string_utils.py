def multiple_replace(string):
    if string == 'TE':
        return string
    string=string.replace('count_tracts','ct')
    string=string.replace('count','c')
    string=string.replace('mismatch','m')
    string=string.replace('utr_3','3utr')   
    string=string.replace('utr_5','5utr')  
    string=string.replace('optimal','opt')
    string=string.replace('m_2','m2')
    string=string.replace('m_0','m0')
    string=string.replace('m_0','m0')
    string=string.replace('T','U')
    return string
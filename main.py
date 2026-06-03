from LaxWendroff import*
from tracer import *
from donnee import *

data1 = Donnee1D()                       #Lax Wendroff
data2 = Donnee1D()                       #Solution analytique

U = LaxWendroff1D(data1)
#anim1D(data1)

u = analytique1D(data2)
anim1D(data2)

analytiqueVSnumerique(data1, data2)
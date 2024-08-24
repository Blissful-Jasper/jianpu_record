# Makefile for HISD sample
# Specify the path according to the execution environment.
CC = /usr/bin/gcc
INC = -I/usr/local/include -I/Software/netcdf-c-4.6.3/include/
LIB = -L/usr/local/lib64 -L/Software/netcdf-c-4.6.3/lib/ -lm -lnetcdf -lz  

MODULE = hisd2netcdf

SRC = main.c hisd_pixlin2lonlat.c hisd_read.c date_utl.c
OBJ = $(SRC:%.c=%.o)

$(MODULE) : $(OBJ)
	$(CC) -o  $@ $(OBJ) $(LIB)

.c.o:
	$(CC) -c -Wall $(INC) $<

clean:
	-rm -f *.o $(MODULE)

 main.o					:hisd.h
 hisd_pixlin2lonlat.o   :hisd.h
 hisd_read.o            :hisd.h
 date_utl.o             :date_utl.h
 

CXX      ?= mpic++
CXXFLAGS ?= -std=c++20 -fopenmp
CPPFLAGS ?= -g -Wall -pedantic -I. -I$(PACS_ROOT)/include -I ../test -I${mkEigenInc}
LDFLAGS  ?= -L$(PACS_ROOT)/src/Utities -L$(PACS_ROOT)/lib
LDLIBS   ?= -lmuparser -lmpi 
LINK.o := $(LINK.cc) # Use C++ linker.

DEPEND = make.dep

EXEC = main
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)

.PHONY = all $(EXEC) $(OBJS) clean distclean $(DEPEND)

all: $(DEPEND) $(EXEC)

$(EXEC): $(OBJS)

$(OBJS): %.o: %.cpp

clean:
	$(RM) $(DEPEND)
	$(RM) *.o

distclean: clean
	$(RM) $(EXEC)
	$(RM) *.csv *.out *.bak *~

$(DEPEND): $(SRCS)
	@$(RM) $(DEPEND)
	@for file in $(SRCS); \
	do \
	  $(CXX) $(CPPFLAGS) $(CXXFLAGS) -MM $${file} >> $(DEPEND); \
	done

-include $(DEPEND)

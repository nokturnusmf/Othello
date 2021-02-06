CXX := g++ -std=c++1z -pthread
CXX_FLAGS := -Wall -Wextra -Wno-sign-compare -O3 -march=native
CUC := nvcc -O3
CUC_FLAGS := $(foreach option,$(CXX_FLAGS),-Xcompiler $(option))

LIBS := -lcudnn

COMMON_SRC   := board tree mcts stop neural
SELFPLAY_SRC := self_play self_play_data $(COMMON_SRC)
ANALYZE_SRC  := analyze $(COMMON_SRC)
AUGMENT_SRC  := augment board self_play_data
TESTING_SRC  := testing $(COMMON_SRC)

SELFPLAY_OBJ := $(patsubst %,bin/obj/%.o,$(SELFPLAY_SRC))
ANALYZE_OBJ  := $(patsubst %,bin/obj/%.o,$(ANALYZE_SRC))
AUGMENT_OBJ  := $(patsubst %,bin/obj/%.o,$(AUGMENT_SRC))
TESTING_OBJ  := $(patsubst %,bin/obj/%.o,$(TESTING_SRC))

all: bin/self_play bin/analyze bin/augment bin/testing

bin/self_play: $(SELFPLAY_OBJ)
	$(CUC) -o $@ $^ $(CUC_FLAGS) $(LIBS)

bin/analyze: $(ANALYZE_OBJ)
	$(CUC) -o $@ $^ $(CUC_FLAGS) $(LIBS)

bin/augment: $(AUGMENT_OBJ)
	$(CXX) -o $@ $^ $(CXX_FLAGS)

bin/testing: $(TESTING_OBJ)
	$(CUC) -o $@ $^ $(CUC_FLAGS) $(LIBS)

bin/obj/%.o: src/%.cpp | bin
	$(CXX) -c -MMD -o $@ $< $(CXX_FLAGS)

bin/obj/%.o: src/%.cu | bin
	$(CUC) -c -MMD -o $@ $< $(CUC_FLAGS)

clean:
	find bin -type f -delete

bin:
	mkdir -p bin/obj

-include $(wildcard bin/obj/*.d)

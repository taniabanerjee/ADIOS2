#include <assert.h>
#include <vector>
#include <algorithm>

typedef std::size_t                  SizeT;
typedef SizeT                        CountType;
typedef std::pair<double, CountType> ValueCountPair;
typedef std::vector<double>          LimitsContainer;
typedef std::vector<ValueCountPair>  ValueCountPairContainer;

// helper funcs
template <typename T> T Min(T a, T b) { return (a<b) ? a : b; }

void ClassifyJenksFisherFromValueCountPairs(LimitsContainer& breaksArray, SizeT k, const ValueCountPairContainer& vcpc);
void GetValueCountPairs(ValueCountPairContainer& vcpc, const double* values, SizeT n);

#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace std;
using namespace arma;


class getDataArray {
public:
  Encoder(CharacterVector character_, List sequence_) :
    character(character_), sequence(sequence_) {}

  mat makeMatrix() {
    return mat matrix(sequence.size(), character.size());
  }

  mat coder(Character character, List sequence) {
    RNGScope scope;
    return sapply(input_character, function(x) { as.integer(x == input_sequence[[i]])});
  }
  
  CharacterVector character;
  List sequence;
};


cube fillData() {
  
}


RCPP_MODULE(getDataArray) {

  class_<getDataArray>("getDataArray")
  
  .constructor<CharacterVector, List>
  
  .method("getData", &)
}

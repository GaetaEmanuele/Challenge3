#include <muParser.h>

#include <memory>
#include <string>
#include <cmath>

class MuparserFun
{
public:

  MuparserFun(const std::string &s)
  {
    try
      {        
        initializeParser();
        m_parser.DefineVar("x", &m_var);
        m_parser.DefineVar("y", &m_var2);
        m_parser.SetExpr(s);
      }
    catch (mu::Parser::exception_type &e)
      {
        std::cerr << e.GetMsg() << std::endl;
      }
  };

  double
  operator()(const double &x,const double&y)
  {
    m_var = x;
    m_var2 = y;
    double res = m_parser.Eval();
    return res;
  };

private:
  double     m_var;
  double     m_var2;
  mu::Parser m_parser;
  void initializeParser()
    {
        // Define sin and cos in the parser
        m_parser.DefineFun("sin", [](double x) -> double {
                                  return std::sin(x);
                                  });
        m_parser.DefineFun("cos", [](double x) -> double {
                                  return std::cos(x);
                                  });
        // Define pi in the parser
        m_parser.DefineConst("pi", M_PI); 
    }
};
---
layout: post
title: 词法 flex 语法 bison 分析器使用
categories: 编译原理
---

## flex

```lex
// tokens.l

%{
#include <string>
#include "ast.hpp"
#include "parser.hpp"
#define STRING_TOKEN        yylval.string = new std::string(yytext, yyleng)
#define KEYWORD_TOKEN(t)    yylval.token = t
%}
%option noyywrap
%option header-file="tokens.hpp"

/* Definitions, note: \042 is '"' */
INTEGER              ([0-9]+)
IDENTIFIER           (%[0-9]+)
ATTRIBUTE            ([_a-zA-Z][_a-zA-Z0-9]*)

%%
[\n]                 ;
[ \t\r\a]+           ;

 /* Symbol */
"="                  KEYWORD_TOKEN(T_eq); return T_eq;
"."                  KEYWORD_TOKEN(T_dot); return T_dot;
":"                  KEYWORD_TOKEN(T_colon); return T_colon;
"["                  KEYWORD_TOKEN(T_leftSquareBracket); return T_leftSquareBracket;
"]"                  KEYWORD_TOKEN(T_rightSquareBracket); return T_rightSquareBracket;
"("                  KEYWORD_TOKEN(T_leftBracket); return T_leftBracket;
")"                  KEYWORD_TOKEN(T_rightBracket); return T_rightBracket;
","                  KEYWORD_TOKEN(T_comma); return T_comma;

".dealloc"           STRING_TOKEN; return T_dealloc;

{INTEGER}            STRING_TOKEN; return T_IntConstant;
{IDENTIFIER}         STRING_TOKEN; return T_Identifier;
{ATTRIBUTE}          STRING_TOKEN; return T_Attribute;

<<EOF>>              return 0;
.                    printf("Unknown token!\n"); yyterminate();
%%
```

## Bison

AST 的结构：
![20230104101244](https://cdn.jsdelivr.net/gh/kexve/img@main/image_blog20230104101244.png)
对应的 cpp 代码

```c++
#ifndef AST_H
#define AST_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>

// 所有 AST 的基类
class BaseAST
{
  public:
    virtual ~BaseAST() = default;

    virtual void Dump() const = 0;
};

// CompUnit 是 BaseAST
class CompUnitAST : public BaseAST
{
  public:
    // 用智能指针管理对象
    std::unique_ptr<BaseAST> graph_decl;

    void Dump() const override
    {
        std::cout << "CompUnitAST { ";
        if (graph_decl)
        {
            graph_decl->Dump();
        }
        std::cout << " }";
    }
};

class GraphDeclAST : public BaseAST
{
  public:
    std::unique_ptr<BaseAST> method_call;
    std::unique_ptr<BaseAST> assignmentList;
    std::unique_ptr<BaseAST> return_stmt;
    void Dump() const override
    {
        std::cout << "GraphDeclAST { ";
        if (method_call)
        {
            method_call->Dump();
        }
        if (assignmentList)
        {
            assignmentList->Dump();
        }
        if (return_stmt)
        {
            return_stmt->Dump();
        }
        std::cout << " }";
    }
};

// FuncDef 也是 BaseAST
class MethodCallAST : public BaseAST
{
  public:
    std::unique_ptr<BaseAST> variable_list;

    void Dump() const override
    {
        std::cout << "MethodCallAST { ";
        if (variable_list)
        {
            variable_list->Dump();
        }
        std::cout << " }";
    }
};

class VariableListAST : public BaseAST
{
  public:
    std::vector<std::unique_ptr<BaseAST>> variable;

    void Dump() const override
    {
        std::cout << "VariableListAST { ";
        for (auto iter = variable.begin(); iter != variable.end(); iter++)
        {
            (*iter)->Dump();
        }
        std::cout << " }";
    }
};

class VariableAST : public BaseAST
{
  public:
    // int value_type;
    std::string *ident;
    std::unique_ptr<BaseAST> value_type;

    void Dump() const override
    {
        std::cout << "VariableAST { ";
        // std::cout << "value type: " << value_type << " ";
        std::cout << " ident: " << *ident << " ";
        if (value_type)
        {
            value_type->Dump();
        }
        std::cout << " }";
    }
};

class ValueTypeAST : public BaseAST
{
  public:
    std::unique_ptr<BaseAST> martrix_args;
    std::string *value_type;

    void Dump() const override
    {
        std::cout << "ValueTypeAST { ";
        // std::cout << "value type: " << value_type << " ";
        std::cout << " value type: " << *value_type << " ";
        if (martrix_args)
        {
            martrix_args->Dump();
        }
        std::cout << " }";
    }
};

class MatrixArgsAST : public BaseAST
{

  public:
    std::unique_ptr<BaseAST> lists;
    void Dump() const override
    {
        std::cout << "MatrixArgsAST { ";
        if (lists)
        {
            lists->Dump();
        }
        std::cout << " }";
    }
};

class IntListAST : public BaseAST
{

  public:
    std::vector<int> listInt;
    void Dump() const override
    {
        std::cout << "IntListAST { ";
        for (auto iter = listInt.begin(); iter != listInt.end(); iter++)
        {
            std::cout << *iter << " ";
        }
        std::cout << " }";
    }
};

class AssignmentListAST : public BaseAST
{
  public:
    std::vector<std::unique_ptr<BaseAST>> ass_list;
    void Dump() const override
    {
        std::cout << "AssignmentListAST { ";
        for (auto iter = ass_list.begin(); iter != ass_list.end(); iter++)
        {
            (**iter).Dump();
        }
        std::cout << " }";
    }
};

class AssignmentAST : public BaseAST
{
  public:
    std::unique_ptr<BaseAST> var_decl;
    std::string *namespace_;
    std::string *opDecl;
    std::unique_ptr<BaseAST> attribute;
    std::unique_ptr<BaseAST> valuesList;
    void Dump() const override
    {
        std::cout << "AssignmentAST { ";
        if (var_decl)
        {
            var_decl->Dump();
        }
        std::cout << " namespace: " << *namespace_ << " ";
        std::cout << " opDecl: " << *opDecl << " ";
        if (attribute)
        {
            attribute->Dump();
        }
        if (valuesList)
        {
            valuesList->Dump();
        }
        std::cout << " }";
    }
};

class AttributeAST : public BaseAST
{
  public:
    // int constant_;
    std::string *attrName;
    std::string *constant_;
    void Dump() const override
    {
        std::cout << "AttributeAST { ";
        if(attrName){
            std::cout << " attrName: " << *attrName << " ";
        }
        if (constant_)
        {
            std::cout << " constant: " << *constant_ << " ";
        }
        std::cout << " }";
    }
};

class ValuesListAST : public BaseAST
{
  public:
    std::vector<std::string *> value_list;
    void Dump() const override
    {
        std::cout << "ValuesListAST { ";
        for (auto iter = value_list.begin(); iter != value_list.end(); iter++)
        {
            std::cout << **iter << " ";
        }
        std::cout << " }";
    }
};

class ReturnStmtAST : public BaseAST
{
  public:
    std::string *return_value;
    void Dump() const override
    {
        std::cout << "ReturnStmtAST { ";
        std::cout << *return_value;
        std::cout << " }";
    }
};

#endif
```

使用 Bison 定义 BNF 语法范式：

```bison
// parser.y

%code requires {
#include <memory>
#include <string>
}

%{
#include "ast.hpp"
extern int yylex();
// void yyerror(const char *s) { std::printf("Error: %s\n", s);std::exit(1); }
void yyerror(std::unique_ptr<BaseAST> &ast, const char *s);
%}

%parse-param { std::unique_ptr<BaseAST> &ast }

%union {
  BaseAST *ast_val;
  std::string *string;
  int token;
}

/* Define our terminal symbols (tokens). This should
   match our tokens.l lex file. We also define the node type
   they represent.
 */

%token <string> T_Identifier  T_Attribute T_IntConstant T_dealloc
%token <token> T_eq T_dot T_colon T_leftSquareBracket T_rightSquareBracket T_leftBracket T_rightBracket T_comma

/* Define the type of node our nonterminal symbols represent.
   The types refer to the %union declaration above. Ex: when
   we call an ident (defined by union type ident) we are really
   calling an (NIdentifier*). It makes the compiler happy.
 */
%type <ast_val> CompUnit graphDecl methodCall methodCallArgs varDecl matrix_args value_type int_list assignmentList assignment attribute valuesList returnStmt
%type <string> opDecl


%start CompUnit

%%
CompUnit : graphDecl {
         auto comp_unit = std::make_unique<CompUnitAST>();
         comp_unit->graph_decl = std::unique_ptr<BaseAST>($1);
         ast = move(comp_unit);
         }
         ;

graphDecl :  methodCall assignmentList returnStmt { auto ast = new GraphDeclAST(); ast->method_call = std::unique_ptr<BaseAST>($1); ast->assignmentList = std::unique_ptr<BaseAST>($2); ast->return_stmt = std::unique_ptr<BaseAST>($3); $$ = ast; }
         ;

methodCall: T_Attribute T_leftBracket methodCallArgs T_rightBracket T_colon {
         auto ast = new MethodCallAST();
         ast->variable_list = std::unique_ptr<BaseAST>($3);
         $$ = ast;
         }
         ;
methodCallArgs : /*blank*/ { auto ast = new VariableListAST(); $$ = ast; }
         | varDecl {auto ast = new VariableListAST(); ast->variable.push_back(std::unique_ptr<BaseAST>($1)); $$ = ast;}
         | methodCallArgs T_comma varDecl { dynamic_cast<VariableListAST*>($1)->variable.push_back(std::unique_ptr<BaseAST>($3)); }
         ;
varDecl : T_Identifier T_colon value_type { auto ast = new VariableAST(); ast->ident = $1; ast->value_type = std::unique_ptr<BaseAST>($3); $$ = ast; }
         ;
value_type : T_Attribute { auto ast = new ValueTypeAST(); ast->value_type = $1; $$ = ast; }
         | value_type matrix_args { dynamic_cast<ValueTypeAST*>($1)->martrix_args = std::unique_ptr<BaseAST>($2); }
         ;
matrix_args : T_leftBracket int_list T_rightBracket {auto ast = new MatrixArgsAST(); ast->lists = std::unique_ptr<BaseAST>($2); $$ = ast; }
         ;
int_list : T_IntConstant { auto ast = new IntListAST(); ast->listInt.push_back(std::stoi(*$1)); $$ = ast;}
         | int_list T_comma T_IntConstant { dynamic_cast<IntListAST*>($1)->listInt.push_back(std::stoi(*$3)); }
         ;

assignmentList : assignment { auto ast = new AssignmentListAST(); ast->ass_list.push_back(std::unique_ptr<BaseAST>($1)); $$ = ast; }
         | assignmentList assignment { dynamic_cast<AssignmentListAST*>($1)->ass_list.push_back(std::unique_ptr<BaseAST>($2)); }
         ;
assignment : varDecl T_eq T_Attribute T_dot opDecl attribute T_leftBracket valuesList T_rightBracket {auto ast = new AssignmentAST(); ast->var_decl = std::unique_ptr<BaseAST>($1); ast->namespace_ = $3; ast->opDecl = $5; ast->attribute = std::unique_ptr<BaseAST>($6); ast->valuesList = std::unique_ptr<BaseAST>($8); $$=ast; }
         ;
opDecl : T_Attribute
         | T_dealloc
         ;
attribute: /*blank */ { auto ast = new AttributeAST(); $$ = ast; }
         | T_leftSquareBracket T_Attribute T_eq  T_IntConstant T_rightSquareBracket { auto ast = new AttributeAST(); ast->attrName = $2; ast->constant_ = $4; $$ = ast; }
         ;
valuesList: /*blank*/ { auto ast = new ValuesListAST(); $$ = ast; }
         | T_Identifier { auto ast = new ValuesListAST(); ast->value_list.push_back($1); $$ = ast; }
         | valuesList T_comma T_Identifier { dynamic_cast<ValuesListAST*>($1)->value_list.push_back($3); }
         ;

returnStmt : T_Attribute T_dot T_Attribute T_leftBracket T_Identifier T_rightBracket {auto ast = new ReturnStmtAST(); ast->return_value = $5; $$ = ast;}
         ;

%%

void yyerror(std::unique_ptr<BaseAST> &ast, const char *s){
  std::cout << "error: " << s << std::endl;
}
```

## 生成词法扫描 cpp 文件和语法解析 cpp 文件

```sh
bison -d -o parser.cpp parser.y
lex -o tokens.cpp tokens.l
```

会生成 tokens.hpp，tokens.cpp，parser.hpp，parser.cpp 四个文件。

## 组装 AST

```c++
#include <cassert>
#include <iostream>
#include <memory>
#include <string>

#include <ast.hpp>
#include <tokens.hpp>
#include <parser.hpp>

using namespace std;

int main(int argc, char **argv)
{
    // parse input file
    unique_ptr<BaseAST> ast;
    auto ret = yyparse(ast);
    assert(!ret);

    // parse input string
    YY_BUFFER_STATE buffer = yy_scan_string(input);
    std::unique_ptr<BaseAST> ast;
    static_cast<void>(yyparse(ast));
    yy_delete_buffer(buffer);

    // dump AST
    ast->Dump();
    cout << endl;
    return 0;
}
```

## 参考链接

1. [动手写个玩具编译器](https://jeremyxu2010.github.io/2020/10/%E5%8A%A8%E6%89%8B%E5%86%99%E4%B8%AA%E7%8E%A9%E5%85%B7%E7%BC%96%E8%AF%91%E5%99%A8/)

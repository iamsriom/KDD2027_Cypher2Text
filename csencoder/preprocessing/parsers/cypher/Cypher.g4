grammar Cypher;

oC_Cypher : oC_Statement (SP? ';')? EOF ;

oC_Statement : oC_Query ;

oC_Query : oC_RegularQuery ;

oC_RegularQuery : oC_SingleQuery ( SP? oC_Union )* ;

oC_Union : ( UNION SP? oC_SingleQuery )+ ;

oC_SingleQuery : oC_SinglePartQuery | oC_MultiPartQuery ;

oC_SinglePartQuery : ( oC_ReadingClause SP? )* oC_Return ;

oC_MultiPartQuery : ( oC_ReadingClause SP? )* ( oC_UpdatingClause SP? )* oC_With SP? ( oC_ReadingClause SP? )* ( oC_UpdatingClause SP? )* oC_Return ;

oC_UpdatingClause : oC_Create | oC_Merge | oC_Delete | oC_Set | oC_Remove ;

oC_ReadingClause : oC_Match | oC_Unwind | oC_InQueryCall ;

oC_Match : ( OPTIONAL SP? )? MATCH SP? oC_Pattern ( SP? oC_Where )? ;

oC_Unwind : UNWIND SP? oC_Expression SP AS SP oC_Variable ;

oC_Merge : MERGE SP? oC_PatternPart ( SP? oC_MergeAction )* ;

oC_MergeAction : ( ON SP MATCH SP? oC_Set ) | ( ON SP CREATE SP? oC_Set ) ;

oC_Create : CREATE SP? oC_Pattern ;

oC_Set : SET SP? oC_SetItem ( SP? ',' SP? oC_SetItem )* ;

oC_SetItem : ( oC_PropertyExpression SP? '=' SP? oC_Expression ) | ( oC_Variable SP? '=' SP? oC_Expression ) | ( oC_Variable SP? '+=' SP? oC_Expression ) | ( oC_NodeLabels ) ;

oC_Delete : ( DELETE SP? oC_Expression ( SP? ',' SP? oC_Expression )* ) | ( DETACH SP DELETE SP? oC_Expression ( SP? ',' SP? oC_Expression )* ) ;

oC_Remove : REMOVE SP? oC_RemoveItem ( SP? ',' SP? oC_RemoveItem )* ;

oC_RemoveItem : ( oC_Variable SP? oC_NodeLabels ) | oC_PropertyExpression ;

oC_InQueryCall : CALL SP oC_ExplicitProcedureInvocation ( SP? YIELD SP? oC_YieldItems )? ;

oC_ExplicitProcedureInvocation : oC_ProcedureName SP? '(' SP? ( oC_Expression ( SP? ',' SP? oC_Expression )* )? SP? ')' ;

oC_ProcedureName : ( oC_Namespace oC_SymbolicName ) | oC_SymbolicName ;

oC_YieldItems : ( oC_YieldItem ( SP? ',' SP? oC_YieldItem )* ) | '*' ;

oC_YieldItem : ( oC_ProcedureResult SP AS SP oC_Variable ) | oC_ProcedureResult ;

oC_ProcedureResult : oC_PropertyExpression ;

oC_Return : RETURN SP? ( DISTINCT SP? )? oC_ReturnBody ( SP? oC_Order )? ( SP? oC_Skip )? ( SP? oC_Limit )? ;

oC_ReturnBody : oC_ReturnItems ( SP? oC_Where )? ;

oC_ReturnItems : ( '*' ( SP? ',' SP? oC_ReturnItem )* ) | ( oC_ReturnItem ( SP? ',' SP? oC_ReturnItem )* ) ;

oC_ReturnItem : ( oC_Expression SP AS SP oC_Variable ) | oC_Expression ;

oC_Order : ORDER SP BY SP? oC_SortItem ( SP? ',' SP? oC_SortItem )* ;

oC_SortItem : oC_Expression ( SP? ( ASCENDING | ASC | DESCENDING | DESC ) )? ;

oC_Skip : L_SKIP SP? oC_Expression ;

oC_Limit : LIMIT SP? oC_Expression ;

oC_Where : WHERE SP? oC_Expression ;

oC_Pattern : oC_PatternPart ( SP? ',' SP? oC_PatternPart )* ;

oC_PatternPart : ( oC_Variable SP? '=' SP? oC_AnonymousPatternPart ) | oC_AnonymousPatternPart ;

oC_AnonymousPatternPart : oC_PatternElement ;

oC_PatternElement : ( oC_NodePattern ( SP? oC_PatternElementChain )* ) | ( '(' oC_PatternElement ')' ( SP? oC_PatternElementChain )* ) ;

oC_NodePattern : '(' SP? ( oC_Variable SP? )? ( oC_NodeLabels SP? )? ( oC_Properties SP? )? ')' ;

oC_PatternElementChain : oC_RelationshipPattern SP? oC_NodePattern ;

oC_RelationshipPattern : ( oC_LeftArrowHead SP? oC_Dash SP? oC_RelationshipDetail? SP? oC_Dash SP? oC_RightArrowHead ) | ( oC_LeftArrowHead SP? oC_Dash SP? oC_RelationshipDetail? SP? oC_Dash ) | ( oC_Dash SP? oC_RelationshipDetail? SP? oC_Dash SP? oC_RightArrowHead ) | ( oC_Dash SP? oC_RelationshipDetail? SP? oC_Dash ) ;

oC_RelationshipDetail : '[' SP? ( oC_Variable SP? )? ( oC_RelationshipTypes SP? )? ( oC_RangeLiteral SP? )? ( oC_Properties SP? )? ']' ;

oC_Properties : oC_MapLiteral | oC_Parameter ;

oC_RelationshipTypes : ':' SP? oC_RelTypeName ( SP? '|' SP? ':'? SP? oC_RelTypeName )* ;

oC_NodeLabels : oC_NodeLabel ( SP? oC_NodeLabel )* ;

oC_NodeLabel : ':' SP? oC_LabelName ;

oC_RangeLiteral : '*' SP? ( oC_IntegerLiteral SP? )? ( '..' SP? oC_IntegerLiteral SP? )? ;

oC_LabelName : oC_SchemaName ;

oC_RelTypeName : oC_SchemaName ;

oC_Expression : oC_OrExpression ;

oC_OrExpression : oC_XorExpression ( SP? OR SP? oC_XorExpression )* ;

oC_XorExpression : oC_AndExpression ( SP? XOR SP? oC_AndExpression )* ;

oC_AndExpression : oC_NotExpression ( SP? AND SP? oC_NotExpression )* ;

oC_NotExpression : ( NOT SP? )? oC_ComparisonExpression ;

oC_ComparisonExpression : oC_StringListNullPredicateExpression ( SP? oC_PartialComparisonExpression )* ;

oC_PartialComparisonExpression : ( '=' SP? oC_StringListNullPredicateExpression ) | ( '<>' SP? oC_StringListNullPredicateExpression ) | ( '<' SP? oC_StringListNullPredicateExpression ) | ( '>' SP? oC_StringListNullPredicateExpression ) | ( '<=' SP? oC_StringListNullPredicateExpression ) | ( '>=' SP? oC_StringListNullPredicateExpression ) ;

oC_StringListNullPredicateExpression : oC_AddOrSubtractExpression ( oC_StringListNullPredicateExpressionPart )* ;

oC_StringListNullPredicateExpressionPart : ( oC_StringPredicateExpression ) | ( oC_ListPredicateExpression ) | ( oC_NullPredicateExpression ) ;

oC_StringPredicateExpression : ( SP? oC_StringOperator SP? oC_AddOrSubtractExpression ) | ( SP? oC_StringOperator SP? oC_RegularExpression ) ;

oC_StringOperator : ( STARTS SP WITH ) | ( ENDS SP WITH ) | ( CONTAINS ) ;

oC_RegularExpression : oC_AddOrSubtractExpression ;

oC_ListPredicateExpression : SP? oC_ListOperator SP? oC_AddOrSubtractExpression ;

oC_ListOperator : IN | oC_StringOperator ;

oC_NullPredicateExpression : SP? ( IS SP NULL ) | ( IS SP NOT SP NULL ) ;

oC_AddOrSubtractExpression : oC_MultiplyDivideModuloExpression ( ( SP? '+' SP? oC_MultiplyDivideModuloExpression ) | ( SP? '-' SP? oC_MultiplyDivideModuloExpression ) )* ;

oC_MultiplyDivideModuloExpression : oC_PowerOfExpression ( ( SP? '*' SP? oC_PowerOfExpression ) | ( SP? '/' SP? oC_PowerOfExpression ) | ( SP? '%' SP? oC_PowerOfExpression ) )* ;

oC_PowerOfExpression : oC_UnaryAddOrSubtractExpression ( SP? '^' SP? oC_UnaryAddOrSubtractExpression )* ;

oC_UnaryAddOrSubtractExpression : ( ( '+' | '-' ) SP? )* oC_PropertyOrLabelsExpression ;

oC_PropertyOrLabelsExpression : oC_Atom ( oC_PropertyLookup | oC_NodeLabels )* ;

oC_Atom : oC_Literal | oC_Parameter | oC_CaseExpression | ( COUNT SP? '(' SP? '*' SP? ')' ) | oC_ListComprehension | oC_PatternComprehension | ( ALL SP? '(' SP? oC_FilterExpression SP? ')' ) | ( ANY SP? '(' SP? oC_FilterExpression SP? ')' ) | ( NONE SP? '(' SP? oC_FilterExpression SP? ')' ) | ( SINGLE SP? '(' SP? oC_FilterExpression SP? ')' ) | oC_RelationshipsPattern | oC_FunctionInvocation | oC_Variable ;

oC_Literal : oC_BooleanLiteral | oC_NumberLiteral | oC_StringLiteral | oC_ListLiteral | oC_MapLiteral ;

oC_BooleanLiteral : TRUE | FALSE ;

oC_NumberLiteral : oC_DoubleLiteral | oC_IntegerLiteral ;

oC_IntegerLiteral : HexInteger | OctalInteger | DecimalInteger ;

oC_DoubleLiteral : RegularDecimalReal | ExponentDecimalReal ;

oC_StringLiteral : ( '"' ( StringLiteral_0 | EscapedChar )* '"' ) | ( '\'' ( StringLiteral_1 | EscapedChar )* '\'' ) ;

oC_ListLiteral : '[' SP? ( oC_Expression ( SP? ',' SP? oC_Expression )* )? SP? ']' ;

oC_MapLiteral : '{' SP? ( oC_PropertyKeyName SP? ':' SP? oC_Expression ( SP? ',' SP? oC_PropertyKeyName SP? ':' SP? oC_Expression )* )? SP? '}' ;

oC_Variable : oC_SymbolicName ;

oC_With : WITH SP? ( DISTINCT SP? )? oC_ReturnBody ( SP? oC_Where )? ( SP? oC_Order )? ( SP? oC_Skip )? ( SP? oC_Limit )? ;

oC_SchemaName : oC_SymbolicName ;

oC_SymbolicName : ( UnescapedSymbolicName | EscapedSymbolicName | HexLetter | COUNT | FILTER | EXTRACT | ANY | NONE | SINGLE ) ;

oC_PropertyKeyName : oC_SchemaName ;

oC_PropertyExpression : oC_Atom ( oC_PropertyLookup )+ ;

oC_PropertyLookup : '.' SP? ( oC_PropertyKeyName ) ;

oC_CaseExpression : ( ( CASE ( SP? oC_CaseAlternatives )+ ) | ( CASE SP? oC_Expression ( SP? oC_CaseAlternatives )+ ) ) ( SP? ELSE SP? oC_Expression )? SP? END ;

oC_CaseAlternatives : WHEN SP? oC_Expression SP? THEN SP? oC_Expression ;

oC_ListComprehension : '[' SP? oC_FilterExpression ( SP? '|' SP? oC_Expression )? SP? ']' ;

oC_PatternComprehension : '[' SP? ( oC_Variable SP? '=' SP? )? oC_RelationshipsPattern SP? ( oC_Where SP? )? ( SP? '|' SP? oC_Expression )? SP? ']' ;

oC_FilterExpression : oC_IdInColl ( SP? oC_Where )? ;

oC_IdInColl : oC_Variable SP IN SP? oC_Expression ;

oC_FunctionInvocation : oC_FunctionName SP? '(' SP? ( DISTINCT SP? )? ( oC_Expression ( SP? ',' SP? oC_Expression )* )? SP? ')' ;

oC_FunctionName : ( oC_Namespace oC_SymbolicName ) | oC_SymbolicName ;

oC_Namespace : ( oC_SymbolicName '.' )+ ;

oC_RelationshipsPattern : oC_NodePattern ( SP? oC_PatternElementChain )+ ;

oC_Parameter : '$' ( oC_SymbolicName | DecimalInteger ) ;

// Tokens
SP : ( WHITESPACE )+ ;

WHITESPACE : ( SPACE | TAB ) ;

fragment SPACE : ' ' ;

fragment TAB : '\t' ;

CYPHER : ( C Y P H E R ) ;

EXPLAIN : ( E X P L A I N ) ;

PROFILE : ( P R O F I L E ) ;

USING : ( U S I N G ) ;

PERIODIC : ( P E R I O D I C ) ;

COMMIT : ( C O M M I T ) ;

UNION : ( U N I O N ) ;

ALL : ( A L L ) ;

OPTIONAL : ( O P T I O N A L ) ;

MATCH : ( M A T C H ) ;

UNWIND : ( U N W I N D ) ;

AS : ( A S ) ;

MERGE : ( M E R G E ) ;

ON : ( O N ) ;

CREATE : ( C R E A T E ) ;

SET : ( S E T ) ;

DELETE : ( D E L E T E ) ;

DETACH : ( D E T A C H ) ;

REMOVE : ( R E M O V E ) ;

WITH : ( W I T H ) ;

RETURN : ( R E T U R N ) ;

DISTINCT : ( D I S T I N C T ) ;

ORDER : ( O R D E R ) ;

BY : ( B Y ) ;

L_SKIP : ( S K I P ) ;

LIMIT : ( L I M I T ) ;

ASCENDING : ( A S C E N D I N G ) | ( A S C ) ;

DESCENDING : ( D E S C E N D I N G ) | ( D E S C ) ;

WHERE : ( W H E R E ) ;

OR : ( O R ) | ( '|' '|' ) ;

XOR : ( X O R ) ;

AND : ( A N D ) | ( '&' '&' ) ;

NOT : ( N O T ) | ( '!' ) ;

IN : ( I N ) ;

STARTS : ( S T A R T S ) ;

ENDS : ( E N D S ) ;

CONTAINS : ( C O N T A I N S ) ;

IS : ( I S ) ;

NULL : ( N U L L ) ;

COUNT : ( C O U N T ) ;

FILTER : ( F I L T E R ) ;

EXTRACT : ( E X T R A C T ) ;

ANY : ( A N Y ) ;

NONE : ( N O N E ) ;

SINGLE : ( S I N G L E ) ;

TRUE : ( T R U E ) ;

FALSE : ( F A L S E ) ;

CASE : ( C A S E ) ;

ELSE : ( E L S E ) ;

END : ( E N D ) ;

WHEN : ( W H E N ) ;

THEN : ( T H E N ) ;

YIELD : ( Y I E L D ) ;

CALL : ( C A L L ) ;

oC_LeftArrowHead : '<' ;

oC_RightArrowHead : '>' ;

oC_Dash : '-' ;

UnescapedSymbolicName : ( IdentifierStart ( IdentifierPart )* ) ;

EscapedSymbolicName : ( '`' ( EscapedSymbolicName_0 )* '`' ) ;

fragment EscapedSymbolicName_0 : ( ~[`\\] | EscapedChar ) ;

HexInteger : ( '0' [Xx] ( HexDigit )+ ) ;

DecimalInteger : ( ( Digit )+ ) ;

OctalInteger : ( '0' ( ( OctDigit )+ ) ) ;

HexLetter : [A-Fa-f] ;

HexDigit : ( Digit | HexLetter ) ;

Digit : [0-9] ;

OctDigit : [0-7] ;

StringLiteral_0 : ~["\\] ;

StringLiteral_1 : ~['\\] ;

EscapedChar : ( '\\' ( [btnfr"'\\] | UnicodeEscapedChar ) ) ;

UnicodeEscapedChar : ( 'u' ( HexDigit HexDigit HexDigit HexDigit ) ) ;

RegularDecimalReal : ( ( Digit )+ '.' ( Digit )+ ) ;

ExponentDecimalReal : ( ( ( Digit )+ | ( ( Digit )+ '.' ( Digit )+ ) | ( '.' ( Digit )+ ) ) [Ee] ( '-' | '+' )? ( Digit )+ ) ;

fragment A : [Aa] ;

fragment B : [Bb] ;

fragment C : [Cc] ;

fragment D : [Dd] ;

fragment E : [Ee] ;

fragment F : [Ff] ;

fragment G : [Gg] ;

fragment H : [Hh] ;

fragment I : [Ii] ;

fragment J : [Jj] ;

fragment K : [Kk] ;

fragment L : [Ll] ;

fragment M : [Mm] ;

fragment N : [Nn] ;

fragment O : [Oo] ;

fragment P : [Pp] ;

fragment Q : [Qq] ;

fragment R : [Rr] ;

fragment S : [Ss] ;

fragment T : [Tt] ;

fragment U : [Uu] ;

fragment V : [Vv] ;

fragment W : [Ww] ;

fragment X : [Xx] ;

fragment Y : [Yy] ;

fragment Z : [Zz] ;

fragment IdentifierStart : [A-Za-z_] ;

fragment IdentifierPart : ( IdentifierStart | [0-9] ) ;

-- CASE WHEN THEN EXPRESSION...(LIKE IN PYTHON WE HAVE IF ELSE STATEMENT).
-- 1) SIMPLE CASE WHEN THEN EXPRESSION
-- 2) SEARCHED CASE WHEN THEN EXPRESSION

-- SIMPLE CASE WHEN THEN EXPRESSION:(WHEN WE APPLY CODITION ON A SINGLE COLUMN I.E KNOWN AS SIMPLE CASE WHEN THEN EXPRESSION...)
SELECT FIRST_NAME,LAST_NAME,JOB_ID,SALARY,
CASE JOB_ID WHEN 'IT_PROG' THEN SALARY +1000
            WHEN 'SA_REP' THEN SALARY +2000
            ELSE SALARY
            END AS NEW_1 
            FROM EMPLOYEES; 

-- SEARCHED CASE EXPRESSION :
SELECT FIRST_NAME,LAST_NAME,JOB_ID,SALARY,
CASE WHEN JOB_ID='IT_PROG' THEN SALARY+5000
     WHEN FIRST_NAME LIKE 'S%' THEN SALARY+5000
     ELSE SALARY
     END AS NEW_1
     FROM EMPLOYEES;


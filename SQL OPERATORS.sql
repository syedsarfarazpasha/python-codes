-- FUNDAMENTALS OF SQL....

USE HR;   
SELECT * FROM EMPLOYEES;  

-- ALTERNATE METHOD FOR ABOVE STATEMENTS..
SELECT * FROM HR.EMPLOYEES;

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES;

-- OPERATORES..

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE SALARY >15000;

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE SALARY BETWEEN 10000 AND 15000;

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE JOB_ID='IT_PROG' AND SALARY>10000;

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE JOB_ID='IT_PROG' OR  SALARY>10000;

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE JOB_ID='IT_PROG' OR  EMPLOYEE_ID=108;

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE JOB_ID IN ('IT_PROG','SA_REP');

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE EMPLOYEE_ID IN (103,108,112);

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE EMPLOYEE_ID NOT IN (103,108,112);

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE JOB_ID  LIKE 'IT%';

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE EMPLOYEE_ID  LIKE '103%';

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE JOB_ID NOT LIKE 'IT%';

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE EMPLOYEE_ID NOT LIKE '103%';

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE FIRST_NAME  LIKE '%S';

SELECT EMPLOYEE_ID, FIRST_NAME,LAST_NAME,JOB_ID,SALARY FROM EMPLOYEES
WHERE FIRST_NAME  LIKE 'S_E%';



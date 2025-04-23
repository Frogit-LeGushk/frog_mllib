try:
    import cx_Oracle
except Exception as err:
    print("pip install cx-Oracle")
    raise err

try:
    import psycopg2
except Exception as err:
    print("pip install psycopg2-binary # Нужно устанавливать бинарную версию пакета (там скомпилировано все за вас)")
    raise err

import pandas as pd
import numpy as np
import polars as pl
import time
import os

import queue
from multiprocessing import Process, Queue, Lock, Value
from functools import partial 


def RETRY(function, count=1, is_stacktrace=False, sleep_between=1):
    if count is None:
        count = -1
    while True:
        count -= 1
        try:
            return function()
        except Exception as err:
            if is_stacktrace:
                traceback.print_exc()
            if count == 0:
                raise err
        finally:
            time.sleep(sleep_between)

def cursor_loader(key, sql, cursor, login):
    """
    Загружает данные по заданному курсору и скрипту.
    В результате возвращает датафрем и одновременно сохраняет по ключу в качестве <key>.parquet.zip файла
    
    Params:
    -------
    dirname: str
        Директория, в которую сохраняются файлы по ключу `key`
    key: str
        Ключ, который уникально идентифицирует запрос
    cursor: OracleConnection2 | GreenplumConnection
        Обертка над стандартным библиотечным модулем
    sql_script: str
    printer: print-like function
        Можно передавать logger или thread-safe реализации функции print
    count_retry: int | None
        Количество попыток для запроса, если произошла ошибка
        None - inf try count
    """
    path = login['directory'] + '/' + key + '.parquet.gzip'
    if os.path.isfile(path):
        login['printer']("Table [" + key + "] exist in disk")
        return pd.read_parquet(path)
    
    login['printer']("Loading table [" + key + "]")
    start = time.time()
    
    get_data = lambda: cursor.execute(sql).fetch()
    data = RETRY(get_data, count=login['count_retry'], sleep_between=login['sleep_between_req'])
    
    login['printer']("[" + key + "] loaded for " + str(time.time() - start))
    
    data.to_parquet(path, compression='gzip', index=False)
    return data

    

class OracleConnection2(object):
    def __init__(self, username, password, database='', dns=None, service_name=None,
                is_cache=True, mem_last_n=None, is_debug=True, printer=print, **kwargs):
        """dns = (ip, port, SID)"""
        self.config = {
            "user": username,
            "password": password,
            "host": database,
            "database": database,
            "dns": dns
        }

        if dns is None:
            assert database != '', "database should not be empty"
            self.con = cx_Oracle.connect(username, password, database, encoding="UTF-8", nencoding="UTF-8")
        else:
            assert dns is not None, "dns should not be empty"
            ip, port, sid = dns
            if service_name:
                dsn_tns = cx_Oracle.makedsn(ip, port, service_name=service_name)
            else:
                dsn_tns = cx_Oracle.makedsn(ip, port, sid)
            self.con = cx_Oracle.connect(username, password, dsn_tns, encoding="UTF-8", nencoding="UTF-8")

        self.cur = self.con.cursor()
        sql = 'ALTER SESSION SET NLS_TERRITORY=RUSSIA'
        self.cur.execute(sql)
        
        self.is_cache = is_cache
        self.mem_last_n = mem_last_n
        self.is_debug = is_debug
        self.prefix = "[ORA " + str(id(self)) + ":" + str(os.getpid()) + "] "
        self.printer = printer
        
        if is_debug:
            if self.config['database'] != '':
                server_name = self.config['host'] + ":" + self.config['database']
            else:
                server_name = str(self.config['dns'])
            debug_str = self.prefix + "Connection with " + server_name + ' established'
            self.printer(debug_str)
            self.start = time.time()
        
        if is_cache:
            self.sql = []
            self.params = []
            self.description = []
            self.columns = []
            self.types = []
            self.raw_data = []
            self.data = []

    def __del__(self):
        self.cur.close()
        if 'con' in self.__dict__.keys():
            self.con.close()
        if self.is_debug:
            if self.config['database'] != '':
                server_name = self.config['host'] + ":" + self.config['database']
            else:
                server_name = str(self.config['dns'])
            debug_str = self.prefix + "Connection with " + server_name + " lost, "
            debug_str += "duration: " + str(time.time() - self.start) + " [s]"
            self.printer(debug_str)

    def execute(self, sql, params={}):
        try:
            self.cur.execute(sql, params)
        except Exception as err:
            if self.is_debug: 
                debug_str = self.prefix + "Exception in execute script:\n" + sql
                self.printer(debug_str)
            raise err
            
        if self.is_cache:
            self.sql.append(sql)
            self.params.append(params)
        return self

    def fetch(self, limit=None, is_chain=False, df=True, 
              strmask=('GID', 'MERCHANT_CATEGORY_CD', 'TRANSACTION_TYPE_CD'),
              clob_to_string=True):
        def get_types(meta_data): return [m[1] for m in meta_data]
        def get_columns(meta_data): return [c[0] for c in meta_data]
        
        def apply_strmask_to_rawdata(data, columns, extended_strmask):
            if any(mask in col for col in columns for mask in extended_strmask):
                def listit(t): # функция конвертации tuple -> list
                    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

                data = listit(data.copy()) # tuple -> list

                # convert GID to str
                for i in filter(lambda col: any(mask in col for mask in extended_strmask), columns):
                    index = columns.index(i)
                    for row in data:
                        row[index] = str(row[index])
            return data
        
        def get_clob_columns(types, columns):
            set_columns = set(strmask)
            for t, c in zip(types, columns):
                if t in [cx_Oracle.DB_TYPE_CLOB, cx_Oracle.LOB]:
                    set_columns.add(c)
            return list(set_columns)

        description = self.cur.description
        columns = get_columns(description)
        types = get_types(description)
        
        try:
            raw_data = self.cur.fetchall() if (limit is None) else self.cur.fetchmany(limit)
            if clob_to_string:
                raw_data = apply_strmask_to_rawdata(raw_data, columns, get_clob_columns(types, columns))
        except Exception as err:
            if self.is_debug: 
                debug_str = self.prefix + "Exception in fetch:"
                self.printer(debug_str)
            raise err
                
        data = pd.DataFrame(raw_data, columns=columns)
        
        # oracle specific typecast (too long numbers)
        for i, (t, c) in enumerate(zip(types, columns)):
            if data.dtypes[c] == 'object' and t == cx_Oracle.NUMBER:
                data[c] = data[c].astype(str)
    
        if self.is_cache:
            self.description.append(description)
            self.columns.append(columns)
            self.types.append(types)
            self.raw_data.append(raw_data)
            self.data.append(data)
            
            if self.mem_last_n is not None:
                self.description = self.description[-self.mem_last_n:]
                self.columns = self.columns[-self.mem_last_n:]
                self.types = self.types[-self.mem_last_n:]
                self.raw_data = self.raw_data[-self.mem_last_n:]
                self.data = self.data[-self.mem_last_n:]
        
        if is_chain: return self
        return data if df else raw_data    
        
    def __create_table__(self, df_table, name):
        sql = 'CREATE TABLE ' + name + ' ('
        for i in df_table.dtypes.items():
            sql = sql + i[0]
            if (i[1].name[:3] == 'int') | (i[1].name[:3] == 'flo'):
                sql = sql + ' NUMBER,'
            elif (i[1].name[:3] == 'dat'):
                sql = sql + ' DATE,'
            else:
                try:
                    maxlen = df_table[i[0]].astype(str).str.len().max() * 2
                except:
                    print(i)
                    return 1

                if str(maxlen) == 'nan': 
                    maxlen = 4
                if str(maxlen) == '0': 
                    maxlen = 1
                if maxlen > 4000:
                    sql = sql + ' CLOB,' 
                else:
                    sql = sql + ' VARCHAR2(' + str(maxlen) + 'byte),'

        sql = sql[:-1] + ')'
        print(sql)
        self.cur.execute(sql)

    def __drop_table__(self, name):
        try:
            sql = 'drop table ' + name + ' purge'
            self.cur.execute(sql)
            print('Table ' + str(name) + ' dropped')
        except:
            pass

    def __fillna__(self, df, fillna):
        if fillna:
            print('\nNaN in float and int replace by -1')
            print("\nNaN in object replace by 'nan'")
            
        
        for col in df.columns:
#             print(fillna)
            if (str(df[col].dtype)[:5] == 'float') | (str(df[col].dtype)[:3] == 'int'):
                if fillna:
                    df[col] = df[col].fillna(-1)
                else:
                    df[col] = df[col].replace({np.nan: None})
            elif (str(df[col].dtype)[:5] == 'objec'):
#                 print(col)
                if fillna:
                    df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(str).replace({'nan': None})

    def __insert_into__(self, df, name):
        sql = 'insert into ' + name + '('
        for i in df.dtypes.items():
            sql = sql + i[0] + ','
        sql = sql[:-1] + ') values ('

        for j, i in enumerate(df.dtypes.items()):
            if i[1].name[:3] == 'dat':
                sql = sql + 'to_date(:' + str(j + 1) + ', \'yyyy-mm-dd HH24:MI:SS\'),'
                df[i[0]] = df[i[0]].map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(x) else '')
            else:
                sql = sql + ':' + str(j + 1) + ','
        sql = sql[:-1] + ')'

        self.cur.prepare(sql)
#         self.cur.setinputsizes()
#         self.cur.executemanyprepared(df.values.tolist())
        self.cur.executemany(None, df.values.tolist())
        self.con.commit()

    def to_oracle(self, df_table, name, create=True, part_size=None, fillna=True):
        if create:
            self.__drop_table__(name)
            self.__create_table__(df_table, name)
        if part_size is None:
            df = df_table.copy()
            self.__fillna__(df, fillna)
            self.__insert_into__(df, name)
        else:
            i = 0
            while True:
                if len(df_table[i * part_size:(i + 1) * part_size]) == 0:
                    break
                print(i)
                df = df_table[i * part_size:(i + 1) * part_size].copy()
                self.__fillna__(df, fillna)

class GreenplumConnection(object):
    def __init__(self, user, password, 
                 host='', database='', 
                 is_cache=True, mem_last_n=None, is_debug=True, printer=print, **kwargs):
        """
        Конструктор для соединения c базой данных Greenplum.
        Connection держится до тех пор, пока не удалится объект.
        Лучше использовать объект на 1 запрос, после чего удалять (see examples), т.к.
        после запроса с ошибкой - перестает работать
        
        Params:
        -------
        user: str
            Имя вашей учетной записи
        password: str
            Пароль от учетной записи
        host: str
            url по которому находится СУБД
        database: str
            Название базы данных. =
        is_cache: bool
            Делать ли кеширование для всех запросов. (можно делать запросы в цикле)
        mem_last_n: None|int
            Если `is_cache == True`, то сколько последних записей сохранять? 
            Если `mem_last_n == None`, то все записи сохраняются
        is_debug: True
            Выводить дебаг печать или нет
        printer: print-like function
            Можно передавать logger или thread-safe реализации функции print
        
        Fields:
        -------
        self.config
        self.con
        self.cur
        self.is_cache
        self.mem_last_n
        self.is_debug
        self.printer = printer
        
        if is_cache:
            self.sql = []
            self.params = []
            self.description = []
            self.columns = []
            self.types = []
            self.raw_data = []
            self.data = []
        """
        self.config = {
            "user": user,
            "password": password,
            "host": host,
            "database": database
        }
        self.con = psycopg2.connect(**self.config)
        self.cur = self.con.cursor()
        self.is_cache = is_cache
        self.mem_last_n = mem_last_n
        self.is_debug = is_debug
        self.prefix = "[GPL " + str(id(self)) + ":" + str(os.getpid()) + "] "
        self.printer = printer
        
        if is_debug:
            debug_str = self.prefix + "Connection with " + self.config['host'] + ":" + self.config['database'] + ' established'
            self.printer(debug_str)
            self.start = time.time()
        
        if is_cache:
            self.sql = []
            self.params = []
            self.description = []
            self.columns = []
            self.types = []
            self.raw_data = []
            self.data = []

    def __del__(self):
        if 'cur' in self.__dict__.keys():
            self.cur.close()
        if 'con' in self.__dict__.keys():
            self.con.close()
        if self.is_debug:
            debug_str = self.prefix + "Connection with " + self.config['host'] + ":" + self.config['database'] + " lost, "
            debug_str += "duration: " + str(time.time() - self.start) + " [s]"
            self.printer(debug_str)
    
    def execute(self, sql, params={}, autocommit=False):
        """
        Внимание, товарищи! 
        Выгрузка происходит на момент вызова данной функции, а не в момент вызова fetch (как в Oracle)
        
        Если connection был установлен в debug режиме, то в отладке будет присутствовать скрипт, который вызвал ошибку
        """
        try:
            self.cur.execute(sql, params)
            if autocommit:
                self.con.commit()
        except Exception as err:
            if self.is_debug: 
                debug_str = self.prefix + "Exception in execute script:\n" + sql
                self.printer(debug_str)
            raise err
            
        if self.is_cache:
            self.sql.append(sql)
            self.params.append(params)
        return self
        
    def fetch(self, limit=None, is_chain=False, df=True, decimal_to_float=False, use_pandas=True):
        """
        Выгружает данные и сохраняет в историю информацию всех запросов за данный connection.
        
        В целях экономии памяти, лучше не кешировать прошлые результаты. 
        
        Params:
        -------
        limit: int | None
            Количество примеров, которые нужно выгрузить
            Если None, то выгружаются все примеры
        is_chain: Bool
            Возвращать self или нет. Полезно, только тогда, когда включено кеширование
        df: Bool
            Возвращать pd.DataFrame или данные в том виде, в котором пришли с запроса
        decimal_to_float: Bool
            Работает только если `df == True`
            Нужно ли преобразовать данные decimal колонок во float
        """
        def get_types(meta_data): return [m[1] for m in meta_data]
        def get_columns(meta_data): return [c[0] for c in meta_data]

        description = self.cur.description
        columns = get_columns(description)
        types = get_types(description)
        
        try:
            raw_data = self.cur.fetchall() if (limit is None) else self.cur.fetchmany(limit)
        except Exception as err:
            if self.is_debug: 
                debug_str = self.prefix + "Exception in fetch:"
                self.printer(debug_str)
            raise err
            
        cast_cols = []
        if decimal_to_float:
            for t, c in zip(types, columns):
                if t == 1700: # type Decimal('...')
                    cast_cols.append(c)

        if use_pandas:
            data = pd.DataFrame(raw_data, columns=columns)
            data = data.astype({c: 'float64' for c in cast_cols})
        else:
            data = pl.DataFrame(raw_data, columns=columns)
            casts = [pl.col(c).cast(pl.float64) if (c in cast_cols) else pl.col(c) for c in data.columns]
            data = data.select(*casts)
        
        if self.is_cache:
            self.description.append(description)
            self.columns.append(columns)
            self.types.append(types)
            self.raw_data.append(raw_data)
            self.data.append(data)
            
            if self.mem_last_n is not None:
                self.description = self.description[-self.mem_last_n:]
                self.columns = self.columns[-self.mem_last_n:]
                self.types = self.types[-self.mem_last_n:]
                self.raw_data = self.raw_data[-self.mem_last_n:]
                self.data = self.data[-self.mem_last_n:]
        
        if is_chain: return self
        return data if df else raw_data   

    def __insert_into__(self, df, name, role="DP_GP_BA_DSR", autocommit=True):
        """
        Периодически работает очень медленно. В официальной документации написано, что `executemany`
        работает не быстрее чем вставка в цикле, через `execute` (что звучит очень странно). 
        
        На данный момент, пока это не пофиксят - лучше пользоваться методом `__insert_batch_into__` 

        `set_role` нужен чтобы работать от группового пользователя.
        """
        assert False, "Use `__insert_batch_into__` instead, it's has broken impl of (executemany -> decode -> decode_utf8) function"

        try:
            sql = "set role " + role
            if role != "":
                self.cur.execute(sql)
            
            sql = f"insert into {name} ({','.join(df.columns)}) values ({','.join(['%s'] * len(df.columns))})"
            
            if isinstance(df, pl.dataframe.frame.DataFrame):
                self.cur.executemany(sql, df.rows())
            else: # pandas case
                self.cur.executemany(sql, df.values.tolist())
                
            if autocommit:
                self.con.commit()
        except Exception as err:
            if self.is_debug: 
                debug_str = self.prefix + "Exception in execute script:\n" + sql
                self.printer(debug_str)
            raise err

        return self

    def __insert_batch_into__(self, batch, name, role="DP_GP_BA_DSR", autocommit=True):
        """
        Более быстрый аналог метода `__insert_into__`
        """
        if role != "":
            self.execute("set role " + role, autocommit=False)
        
        column_names = ','.join([f'"{col}"' for col in batch.columns])
        sql_template = f"insert into {name} ({column_names}) values "
        sql_template += ','.join( ['(' +  ','.join(['%s'] * batch.shape[1]) + ')'] * batch.shape[0]  )

        if isinstance(batch, pl.dataframe.frame.DataFrame):
            sql = self.cur.mogrify(sql_template, batch.to_numpy().flatten())
        else: # pandas case
            sql = self.cur.mogrify(sql_template, batch.values.flatten())
        
        return self.execute(sql.decode("utf-8"), autocommit=autocommit)

    def __create_table__(self, df, table_name, schema_name):
        if isinstance(df, pl.dataframe.frame.DataFrame):
            df = df.to_pandas()
        sql = pd.io.sql.get_schema(df, table_name, schema=schema_name)
        return self.execute(sql, autocommit=True)


class FutureWorkers:
    def __init__(self, processes, counter, n_tasks, directory):
        self.processes_ = processes
        self.counter_ = counter
        self.n_tasks_ = n_tasks
        self.directory_ = directory
        
    def progress(self):
        return f"Progress: {round(self.counter_.value / self.n_tasks_ * 100, 2)} %"
    
    def start(self):
        [p.start() for p in self.processes_]
        time.sleep(1)
        return self
    
    def kill(self):
        [p.kill() for p in self.processes_]
        time.sleep(1)
        return self
    
    def join(self):
        for p in self.processes_:
            try:
                p.join()
            except Exception as err:
                err_text = f"Exception {str(err)}, type {type(err)}"
                print(err_text)
        time.sleep(1)
        return self
    
    def unload_results(self, use_polars=True, make_concatenate=True, **kw_read_parquet):
        """
        Возвращает промежуточный результат загрузки
        `kw_read_parquet` - позволяет передать дополнительные опции чтения из parquet файла
        """
        df_union = []
        for file in os.listdir(self.directory_):
            path = self.directory_ + '/' + file
            if os.path.isfile(path):
                if use_polars:
                    df_union.append(pl.read_parquet(path, **kw_read_parquet))
                else:
                    df_union.append(pd.read_parquet(path, **kw_read_parquet))
                    
        if make_concatenate:
            return pl.concat(df_union) if use_polars else pd.concat(df_union)
        else:
            return df_union
    
def run_workers(
    tasks, 
    logins, 
    dwh_workers, 
    ehd_workers, 
    gpl_workers, 
    directory, 
    count_retry=1, 
    sleep_between_req=1
):
    """
    Запускает параллельно выполняться задачи `tasks` с последующим сохранением в директорию `directory`
    Загрузка может быть произведена с курсорами `OracleConnection2`, `GreenplumConnection`
    Возвращается объект `FutureWorkers` для отслеживания и прекращения процесса загрузки
    """
    q_dwh = Queue()
    q_ehd = Queue()
    q_gpl = Queue()
    counter = Value('i', 0, lock=True)
    logins = logins.copy()
    lock = Lock()
    n_tasks = len(tasks)
    
    def process_safe_printer(*args, is_debug):
        if is_debug:
            lock.acquire()    
            try:
                print(*args, flush=True)
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}", flush=True)
            finally: 
                lock.release()
    
    for key in logins.keys():
        logins[key] = logins[key].copy()
        logins[key]['directory'] = directory
        logins[key]['count_retry'] = count_retry
        logins[key]['sleep_between_req'] = sleep_between_req
        if 'is_debug' not in logins[key].keys():
            logins[key]['is_debug'] = True
        logins[key]['printer'] = partial(
            process_safe_printer, 
            is_debug=logins[key]['is_debug']
        )
    
    for db_type, key, sql in tasks:
        if db_type == 'dwh':
            q_dwh.put((key, sql))
        elif db_type == 'ehd':
            q_ehd.put((key, sql))
        elif db_type == 'gpl':
            q_gpl.put((key, sql))
        else: 
            raise Exception(f"dbtype {db_type} doesn't exist")
            
    if not os.path.isdir(directory):
        os.mkdir(directory)
        
    def worker(q, login, cnt, cursorClass):
        pid = os.getpid()
        login['printer'](f"Worker {cursorClass} [{pid}] started")
        
        while True:
            try:
                cursor = cursorClass(**login)
                data = cursor_loader(*q.get(block=False), cursor, login)
                with cnt.get_lock():
                    cnt.value += 1
            except queue.Empty:
                break
            except Exception as err:
                err_msg = f"Error in key [" + key + f"], error: {str(err)}, type: {type(err)}"
                login['printer'](err_msg)
        
        login['printer'](f"Worker {cursorClass} [{pid}] finished")
            
            
    processes = []
    for _ in range(dwh_workers):
        p = Process(target=worker, args=(q_dwh, logins['dwh'], counter, OracleConnection2))
        processes.append(p)
    for _ in range(ehd_workers):
        p = Process(target=worker, args=(q_ehd, logins['ehd'], counter, OracleConnection2))
        processes.append(p)
    for _ in range(gpl_workers):
        p = Process(target=worker, args=(q_gpl, logins['gpl'], counter, GreenplumConnection))
        processes.append(p)
        
    return FutureWorkers(
        processes, 
        counter, 
        n_tasks, 
        directory
    )

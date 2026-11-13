# 业务资料约定

score为当时资格评分，score_revisions不覆盖原分。received是实际领取，outcome为固定窗口内新增业务量，baseline为政策前业务量；known_day以批次公布日为0。firm是企业相关与首次申请识别键。

每行id在该表内唯一，文件顺序不代表业务事件顺序。清单schema=1，各表登记文件名、字节长度、SHA256、行数及完整字段集合，未知版本或内容变化应重新登记。金额及测量数值均为独立资料，不代表任何现实机构的经营数据。

- policies：id、cutoff、eligible_above、inclusive。
- applications：id、firm、batch、score、received、baseline、outcome、outcome_known_day。
- score_revisions：id、application、new_score、known_day。

normal与boundary是相互独立的资料包，不能直接拼接重复id。引用字段按本项目domain.py的REFERENCES登记。读取检查只检查完整性、身份及这些直接引用，不替代领域分析中的可识别性、时点可见性或可行性判断。

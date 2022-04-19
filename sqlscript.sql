CREATE TABLE samples(
    sample_id        varchar(16) NOT NULL,
    image_id         varchar(16) NOT NULL PRIMARY KEY,
    -- diagnosis        varchar(8)  NOT NULL,
    -- diagnosis_method varchar(16) NOT NULL,
    age              int,
    sex              CHAR(1),
    location         varchar(64) NOT NULL
);
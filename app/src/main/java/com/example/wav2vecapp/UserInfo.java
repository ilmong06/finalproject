package com.example.wav2vecapp;

public class UserInfo {

    public String name;
    public String phnum;
    public String language;
    public String birthdate;
    public String gender;
    public String emergency_name;
    public String emergency_phnum;
    public String emergency_relation;

    public UserInfo(String name, String phnum, String language, String birthdate, String gender,
                    String emergency_name, String emergency_phnum, String emergency_relation) {
        this.name = name;
        this.phnum = phnum;
        this.language = language;
        this.birthdate = birthdate;
        this.gender = gender;
        this.emergency_name = emergency_name;
        this.emergency_phnum = emergency_phnum;
        this.emergency_relation = emergency_relation;
    }
}

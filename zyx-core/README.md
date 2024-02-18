# zyx-core

Core crate of zyx machine learning library. This is used by devices, or if you want to write custom modules
and need to import [Backend] trait.

# Cargo features

std - enables io functions that require filesystem
    - implements std::error::Error for ZyxError


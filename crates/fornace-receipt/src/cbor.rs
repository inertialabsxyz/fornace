//! Deterministic CBOR (RFC 8949, section 4.2.1).
//!
//! ciborium writes integers, lengths, and floats in their shortest form, and
//! all lengths as definite. This module adds the last rule: the keys of each
//! map are in the bytewise order of their encodings.

use ciborium::Value;

/// The deterministic encoding of `value`.
pub(crate) fn encode(value: &Value) -> Vec<u8> {
    write(&canonical(value.clone()))
}

fn write(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::into_writer(value, &mut out).expect("CBOR into a Vec does not fail");
    out
}

fn canonical(value: Value) -> Value {
    match value {
        Value::Map(entries) => {
            let mut keyed: Vec<(Vec<u8>, Value, Value)> = entries
                .into_iter()
                .map(|(k, v)| {
                    let k = canonical(k);
                    (write(&k), k, canonical(v))
                })
                .collect();
            keyed.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Map(keyed.into_iter().map(|(_, k, v)| (k, v)).collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonical).collect()),
        Value::Tag(tag, inner) => Value::Tag(tag, Box::new(canonical(*inner))),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_keys_are_in_the_order_of_their_encodings() {
        // 10 encodes as 0x0a, 265 as 0x19 0x01 0x09, and a text key starts
        // with 0x6_. So the order is 10, 265, "a", "bb".
        let value = Value::Map(vec![
            (Value::Text("bb".into()), Value::Integer(1.into())),
            (Value::Integer(265.into()), Value::Integer(2.into())),
            (Value::Text("a".into()), Value::Integer(3.into())),
            (Value::Integer(10.into()), Value::Integer(4.into())),
        ]);
        assert_eq!(
            encode(&value),
            [
                0xa4, 0x0a, 0x04, 0x19, 0x01, 0x09, 0x02, 0x61, b'a', 0x03, 0x62, b'b', b'b', 0x01
            ]
        );
    }

    #[test]
    fn floats_use_the_shortest_exact_form() {
        // 1.5 is exact in a half float (0xf9), 1.1f32 needs a single (0xfa).
        assert_eq!(encode(&Value::Float(1.5)), [0xf9, 0x3e, 0x00]);
        assert_eq!(encode(&Value::Float(f64::from(1.1f32)))[0], 0xfa);
    }
}

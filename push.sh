#!/bin/sh

echo "=== pushing... ==="
git add --all
echo "Message: "
read message
git commit -m $message
git push origin master
echo "=== over ==="
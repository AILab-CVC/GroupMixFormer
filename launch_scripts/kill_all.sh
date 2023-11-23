flag=test
echo chief ip:  $CHIEF_IP
host_ip_list=(${NODE_IP_LIST//,/ })
let world_size=NODE_NUM/8
YOUR_SCRIP=my
echo world size: $world_size
rank=1
for host in ${host_ip_list[@]}; do
    echo $host
    host_ip=$(echo $host| cut -d':' -f 1)
    echo $host_ip
    if [ $host_ip != $CHIEF_IP ]; then
        ssh -o StrictHostKeyChecking=no -t -f $host_ip "cd /home/chongjian/code/GroupMixFormer; killall python3"
        let rank+=1
    fi
done

echo start chief node
cd /home/chongjian/code/GroupMixFormer; killall python3






